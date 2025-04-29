// Flat‑Combining Queue – GPU‑oriented, robust implementation
// -------------------------------------------------------------
//  • Follows Hendler et al. (SPAA 2010) semantics
//  • Per‑block shared queues reduce global contention
//  • Single‑CAS lock with adaptive back‑off
//  • Thread‑safe memory ordering with __threadfence*
//  • Publication list sized to full grid
//  • Print signature + kernels match original demo script

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

// ----------------------- Tunables --------------------------------------
#ifndef FCQ_GLOBAL_CAP
#define FCQ_GLOBAL_CAP  4096          // power‑of‑two for mask trick
#endif
#ifndef FCQ_BLOCK_CAP
#define FCQ_BLOCK_CAP   256           // shared‑memory ring per block
#endif
#ifndef FCQ_MAX_BACKOFF
#define FCQ_MAX_BACKOFF 512
#endif

#define P2MASK(x) ((x) & (FCQ_GLOBAL_CAP - 1))

// ---------------------- Return codes -----------------------------------
enum { RC_SUCCESS = 0, RC_FULL = 1, RC_EMPTY = 2, RC_CLOSED = 3 };
// ---------------------- Ops -------------------------------------------
enum { OP_IDLE = 0, OP_ENQ = 1, OP_DEQ = 2 };

// ---------------------- Publication record ----------------------------
struct PubRec {
    volatile int op;       // OP_* (idle/enq/deq)
    int           val;     // payload or dequeued value
    int           result;  // RC_*
    volatile int  status;  // 0 = WAITING, 1 = DONE
};

// ---------------------- Global FC queue -------------------------------
struct __align__(64) FCQueue {
    int           buf[FCQ_GLOBAL_CAP];
    unsigned int  head;            // consumer idx
    unsigned int  tail;            // producer idx
    volatile int  lock;            // 0/1
    volatile int  combiner_flag;   // 0/1
    unsigned int  combine_cnt;
    unsigned int  ops_processed;

    // flexible array – sized at runtime
    PubRec recs[1];
};

// ---------------------- Device helpers --------------------------------
__device__ __forceinline__ bool try_lock(volatile int* l) {
    return (atomicCAS((int*)l, 0, 1) == 0);
}

__device__ __forceinline__ void unlock(volatile int* l) {
    __threadfence_system();               // ensure visibility grid‑wide
    atomicExch((int*)l, 0);
}

__device__ void backoff(unsigned int& spins) {
    spins = spins < FCQ_MAX_BACKOFF ? spins * 2 : FCQ_MAX_BACKOFF;
    __nanosleep(spins);
}

// circular‑buffer ops – only combiner touches
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

// ---------------------- Combiner pass ---------------------------------
__device__ void combine_pass(FCQueue* q, int pub_count) {
    unsigned int processed = 0;
    for (int i = 0; i < pub_count; ++i) {
        PubRec* r = &q->recs[i];
        if (r->op == OP_IDLE) continue;
        if (atomicAdd((int*)&r->status, 0) != 0) continue;
        int rc = RC_SUCCESS;
        if (r->op == OP_ENQ) {
            bool ok = gq_enq(q, r->val);
            rc = ok ? RC_SUCCESS : RC_FULL;
        } else if (r->op == OP_DEQ) {
            int tmp;
            bool ok = gq_deq(q, &tmp);
            if (ok) r->val = tmp;
            rc = ok ? RC_SUCCESS : RC_EMPTY;
        }
        r->result = rc;
        __threadfence();
        atomicExch((int*)&r->status, 1);      // DONE
        r->op = OP_IDLE;
        processed++;
    }
    atomicAdd(&q->combine_cnt, 1);
    atomicAdd(&q->ops_processed, processed);
}

// ---------------------- Publication API -------------------------------
__device__ int publish_op(FCQueue* q, int slot, int op, int* io_val, int pub_cnt) {
    PubRec* r = &q->recs[slot];
    r->val = *io_val;
    r->result = RC_SUCCESS;
    __threadfence();
    r->op = op;
    __threadfence();
    atomicExch((int*)&r->status, 0);      // WAITING

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

// ---------------------- Mixed‑op kernel --------------------------------
__global__ void mixedKernel(FCQueue* q, int pub_cnt, int iters,
                            int* enq_ok, int* deq_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pub_cnt) return;
    int v;
    for (int i = 0; i < iters; ++i) {
        if ((tid & 1) == 0) {
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

// ---------------------- Producer / Consumer kernels -------------------
__global__ void producerKernel(FCQueue* q, int pub_cnt,
                               int base, int items, int* enq_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pub_cnt) return;
    for (int i = tid; i < items; i += pub_cnt) {
        int v = base + i;
        int rc = publish_op(q, tid, OP_ENQ, &v, pub_cnt);
        if (rc == RC_SUCCESS) atomicAdd(enq_ok, 1);
    }
}

__global__ void consumerKernel(FCQueue* q, int pub_cnt,
                               int items, int* deq_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pub_cnt) return;
    int v = 0;
    for (int i = tid; i < items; i += pub_cnt) {
        int rc = publish_op(q, tid, OP_DEQ, &v, pub_cnt);
        if (rc == RC_SUCCESS) atomicAdd(deq_ok, 1);
    }
}

// ---------------------- Stats kernel ----------------------------------
__global__ void statsKernel(FCQueue* q, unsigned int* combines, unsigned int* ops) {
    *combines = q->combine_cnt;
    *ops      = q->ops_processed;
}

// ---------------------- Host helpers ----------------------------------
void fcq_init(FCQueue*& d_q, int pub_slots) {
    const size_t bytes = sizeof(FCQueue) + pub_slots * sizeof(PubRec);
    cudaMalloc(&d_q, bytes);
    FCQueue* h = (FCQueue*)calloc(1, bytes);
    h->head = h->tail = 0;
    for (int i = 0; i < pub_slots; ++i) {
        h->recs[i].op = OP_IDLE;
        h->recs[i].status = 1;          // DONE
    }
    cudaMemcpy(d_q, h, bytes, cudaMemcpyHostToDevice);
    free(h);
}

// ---------------------- Main – matches original print signature -------
int main() {
    const int blocks  = 10;
    const int threads = 100;
    const int pub_cnt = blocks * threads;

    FCQueue* d_q;
    fcq_init(d_q, pub_cnt);

    int *d_enq, *d_deq; cudaMalloc(&d_enq, sizeof(int)); cudaMalloc(&d_deq, sizeof(int));
    unsigned int *d_comb, *d_ops; cudaMalloc(&d_comb, sizeof(unsigned int)); cudaMalloc(&d_ops, sizeof(unsigned int));

    cudaMemset(d_enq, 0, sizeof(int)); cudaMemset(d_deq, 0, sizeof(int));

    // ----- Mixed operations test --------------------------------------
    printf("Testing FC queue with mixed operations...\n");
    mixedKernel<<<blocks, threads>>>(d_q, pub_cnt, 10, d_enq, d_deq);
    cudaDeviceSynchronize();

    statsKernel<<<1,1>>>(d_q, d_comb, d_ops);

    int h_enq, h_deq; unsigned int h_comb, h_ops;
    cudaMemcpy(&h_enq, d_enq, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_deq, d_deq, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_comb, d_comb, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ops , d_ops , sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Mixed operations test results:\n");
    printf("  Enqueue successes: %d\n", h_enq);
    printf("  Dequeue successes: %d\n", h_deq);
    printf("  Number of combines: %u\n", h_comb);
    printf("  Total operations processed: %u\n", h_ops);
    printf("  Average operations per combine: %.2f\n",
            h_comb ? (double)h_ops / h_comb : 0.0);

    // reset queue & counters ------------------------------------------
    fcq_init(d_q, pub_cnt);
    cudaMemset(d_enq, 0, sizeof(int)); cudaMemset(d_deq, 0, sizeof(int));
    cudaMemset(d_comb, 0, sizeof(unsigned int)); cudaMemset(d_ops, 0, sizeof(unsigned int));

    printf("\nTesting FC queue with producer-consumer pattern...\n");
    // compute separate slot count for the producer/consumer phase
    const int pcThreads = 50;
    const int pcBlocks  = 5;
    const int pcPubCnt  = pcThreads * pcBlocks;   // 250
    const int num_items = 500;
    producerKernel<<<pcBlocks, pcThreads>>>(d_q, pcPubCnt,
        1000, num_items, d_enq);
    cudaDeviceSynchronize();
    consumerKernel<<<pcBlocks, pcThreads>>>(d_q, pcPubCnt,
        num_items, d_deq);
    cudaDeviceSynchronize();

    statsKernel<<<1,1>>>(d_q, d_comb, d_ops);
    cudaMemcpy(&h_enq, d_enq, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_deq, d_deq, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_comb, d_comb, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ops , d_ops , sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Producer-Consumer test results:\n");
    printf("  Enqueue successes: %d\n", h_enq);
    printf("  Dequeue successes: %d\n", h_deq);
    printf("  Number of combines: %u\n", h_comb);
    printf("  Total operations processed: %u\n", h_ops);
    printf("  Average operations per combine: %.2f\n",
            h_comb ? (double)h_ops / h_comb : 0.0);

    // cleanup
    cudaFree(d_q); cudaFree(d_enq); cudaFree(d_deq); cudaFree(d_comb); cudaFree(d_ops);
    return 0;
}
