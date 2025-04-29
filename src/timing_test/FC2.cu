#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <ctime>

namespace cg = cooperative_groups;

// ---------------------- Configuration ---------------------------------
#ifndef FCQ_GLOBAL_CAP
#define FCQ_GLOBAL_CAP 4096
#endif

#ifndef FCQ_BLOCK_CAP
#define FCQ_BLOCK_CAP 256
#endif

#ifndef FCQ_MAX_BACKOFF
#define FCQ_MAX_BACKOFF 512
#endif

#define P2MASK(x) ((x) & (FCQ_GLOBAL_CAP - 1))

// ---------------------- Return Codes and Operations ------------------
enum { RC_SUCCESS = 0, RC_FULL = 1, RC_EMPTY = 2, RC_CLOSED = 3 };
enum { OP_IDLE = 0, OP_ENQ = 1, OP_DEQ = 2 };

// ---------------------- Structures ------------------------------------
struct PubRec {
    volatile int op;
    int          val;
    int          result;
    volatile int status;
};

struct __align__(64) FCQueue {
    int           buf[FCQ_GLOBAL_CAP];
    unsigned int  head;
    unsigned int  tail;
    volatile int  lock;
    volatile int  combiner_flag;
    unsigned int  combine_cnt;
    unsigned int  ops_processed;
    PubRec        recs[1];
};

// ---------------------- Device Helpers -------------------------------
__device__ __forceinline__ bool try_lock(volatile int* l) {
    return (atomicCAS((int*)l, 0, 1) == 0);
}

__device__ __forceinline__ void unlock(volatile int* l) {
    __threadfence_system();
    atomicExch((int*)l, 0);
}

__device__ void backoff(unsigned int& spins) {
    spins = (spins < FCQ_MAX_BACKOFF) ? spins * 2 : FCQ_MAX_BACKOFF;
    for (volatile unsigned int delay = 0; delay < spins; ++delay) {
        // simple spin-wait
    }
}

// Circular buffer operations (only combiner touches)
__device__ bool gq_enq(FCQueue* q, int v) {
    if ((q->tail - q->head) >= FCQ_GLOBAL_CAP) return false;
    q->buf[P2MASK(q->tail)] = v;
    q->tail++;
    return true;
}

__device__ bool gq_deq(FCQueue* q, int* out) {
    if ((q->tail - q->head) == 0) return false;
    *out = q->buf[P2MASK(q->head)];
    q->head++;
    return true;
}

// ---------------------- Combiner Pass ---------------------------------
__device__ void combine_pass(FCQueue* q, int pub_count) {
    unsigned int processed = 0;
    for (int i = 0; i < pub_count; ++i) {
        PubRec* r = &q->recs[i];
        if (r->op == OP_IDLE || atomicAdd((int*)&r->status, 0) != 0) continue;

        int rc = RC_SUCCESS;
        if (r->op == OP_ENQ) {
            rc = gq_enq(q, r->val) ? RC_SUCCESS : RC_FULL;
        } else if (r->op == OP_DEQ) {
            int tmp;
            rc = gq_deq(q, &tmp) ? (r->val = tmp, RC_SUCCESS) : RC_EMPTY;
        }
        r->result = rc;
        __threadfence();
        atomicExch((int*)&r->status, 1);
        r->op = OP_IDLE;
        processed++;
    }
    atomicAdd(&q->combine_cnt, 1);
    atomicAdd(&q->ops_processed, processed);
}

// ---------------------- Publication API ------------------------------
__device__ int publish_op(FCQueue* q, int slot, int op, int* io_val, int pub_cnt) {
    PubRec* r = &q->recs[slot];
    r->val = *io_val;
    r->result = RC_SUCCESS;
    __threadfence();
    r->op = op;
    __threadfence();
    atomicExch((int*)&r->status, 0);  // WAITING

    unsigned int spin = 32;
    if (try_lock(&q->lock)) {
        q->combiner_flag = 1;
        combine_pass(q, pub_cnt);
        q->combiner_flag = 0;
        unlock(&q->lock);
    }

    while (atomicAdd((int*)&r->status, 0) == 0) {
        if (q->combiner_flag == 0 && try_lock(&q->lock)) {
            q->combiner_flag = 1;
            combine_pass(q, pub_cnt);
            q->combiner_flag = 0;
            unlock(&q->lock);
        } else {
            backoff(spin);
        }
    }
    *io_val = r->val;
    return r->result;
}

// ---------------------- Kernels ---------------------------------------
__global__ void randomMixedKernel(FCQueue* q, int pub_cnt, int iters, int* enq_ok, int* deq_ok, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pub_cnt) return;

    curandState state;
    curand_init(seed, tid, 0, &state);

    int v;
    for (int i = 0; i < iters; ++i) {
        int op = curand(&state) & 1;
        if (op == 0) {
            v = tid * 1000 + i;
            if (publish_op(q, tid, OP_ENQ, &v, pub_cnt) == RC_SUCCESS)
                atomicAdd(enq_ok, 1);
        } else {
            v = 0;
            if (publish_op(q, tid, OP_DEQ, &v, pub_cnt) == RC_SUCCESS)
                atomicAdd(deq_ok, 1);
        }
    }
}

// ---------------------- Host Helpers ----------------------------------
void fcq_init(FCQueue*& d_q, int pub_slots) {
    const size_t bytes = sizeof(FCQueue) + pub_slots * sizeof(PubRec);
    cudaMalloc(&d_q, bytes);

    FCQueue* h = (FCQueue*)calloc(1, bytes);
    h->head = h->tail = 0;
    for (int i = 0; i < pub_slots; ++i) {
        h->recs[i].op = OP_IDLE;
        h->recs[i].status = 1;
    }
    cudaMemcpy(d_q, h, bytes, cudaMemcpyHostToDevice);
    free(h);
}

// ---------------------- Main ------------------------------------------
int main() {
    const int threadsPerBlock = 32;
    const int iters = 1000;

    printf("Blocks, Time(ms), Enqueue Successes, Dequeue Successes\n");

    for (int blocks = 1; blocks <= 49; ++blocks) {
        const int pub_cnt = blocks * threadsPerBlock;

        FCQueue* d_q;
        fcq_init(d_q, pub_cnt);

        int *d_enq, *d_deq;
        unsigned int *d_comb, *d_ops;
        cudaMalloc(&d_enq, sizeof(int));
        cudaMalloc(&d_deq, sizeof(int));
        cudaMalloc(&d_comb, sizeof(unsigned int));
        cudaMalloc(&d_ops, sizeof(unsigned int));

        cudaMemset(d_enq, 0, sizeof(int));
        cudaMemset(d_deq, 0, sizeof(int));
        cudaMemset(d_comb, 0, sizeof(unsigned int));
        cudaMemset(d_ops, 0, sizeof(unsigned int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        unsigned int seed = static_cast<unsigned int>(time(NULL));
        randomMixedKernel<<<blocks, threadsPerBlock>>>(d_q, pub_cnt, iters, d_enq, d_deq, seed);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        int h_enq = 0, h_deq = 0;
        cudaMemcpy(&h_enq, d_enq, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_deq, d_deq, sizeof(int), cudaMemcpyDeviceToHost);

        printf("%d, %.3f, %d, %d\n", blocks, ms, h_enq, h_deq);

        cudaFree(d_q);
        cudaFree(d_enq);
        cudaFree(d_deq);
        cudaFree(d_comb);
        cudaFree(d_ops);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}

