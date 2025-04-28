/**********************************************************************
 *  High-Throughput Blocking Array Queue  32-bit safe (sm_60+)
 *  nvcc -std=c++17 -arch=sm_60 ht_blocking_queue.cu
 *********************************************************************/
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h> // <--- needed for random operation selection

#define QUEUE_SIZE     1024
#define LG_QSIZE       10
#define SLOT_MASK      (QUEUE_SIZE-1)
#define THREADS_PER_BLK 128
#define BLOCKS         10

static_assert((QUEUE_SIZE&(QUEUE_SIZE-1))==0,"size must be power-of-two");

__device__ __forceinline__ uint32_t ld_u32(const volatile uint32_t* p){return *p;}
__device__ __forceinline__ int32_t  ld_i32(const volatile int32_t * p){return *p;}
__device__ __forceinline__ void     full_fence(){ __threadfence(); }
__device__ __forceinline__ void tiny_delay()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(80);
#else
    for(int i=0;i<8;++i) __threadfence_block();
#endif
}
__device__ __host__ __forceinline__
uint32_t gen_of(uint32_t ticket) { return ticket >> LG_QSIZE; }

struct HTQueue {
    volatile uint32_t head, tail;
    volatile int32_t  closed;
    volatile uint32_t data[QUEUE_SIZE];
    volatile uint32_t ids [QUEUE_SIZE];
};

enum { HT_SUCCESS=0, HT_CLOSED=1, HT_BUSY=2 };

__host__ void initQueue(HTQueue* q)
{
    q->head = q->tail = 0;  q->closed = 0;
    for (int i=0;i<QUEUE_SIZE;++i) q->ids[i]=0;
}

__device__ int ht_enqueue(HTQueue* q,uint32_t item)
{
    for(;;){
        if (ld_i32(&q->closed)) return HT_CLOSED;
        uint32_t snap = ld_u32(&q->tail);
        uint32_t idx  = snap & SLOT_MASK;
        uint32_t want = gen_of(snap)*2u;
        if (ld_u32(&q->ids[idx]) != want){ tiny_delay(); continue;}
        if (atomicCAS((unsigned int*)&q->tail,snap,snap+1u)!=snap){
            tiny_delay(); continue;
        }
        q->data[idx]=item; full_fence();
        atomicExch((unsigned int*)&q->ids[idx], want+1u);
        return HT_SUCCESS;
    }
}

__device__ int ht_dequeue(HTQueue* q,uint32_t* out)
{
    for(;;){
        if (ld_i32(&q->closed)) return HT_CLOSED;
        uint32_t snap = ld_u32(&q->head);
        uint32_t idx  = snap & SLOT_MASK;
        uint32_t want = gen_of(snap)*2u+1u;
        if (ld_u32(&q->ids[idx]) != want){ tiny_delay(); continue;}
        if (atomicCAS((unsigned int*)&q->head,snap,snap+1u)!=snap){
            tiny_delay(); continue;
        }
        *out = ld_u32(&q->data[idx]); full_fence();
        atomicExch((unsigned int*)&q->ids[idx], want+1u);
        return HT_SUCCESS;
    }
}

__device__ int ht_enqueue_nb(HTQueue* q,uint32_t item)
{
    if(ld_i32(&q->closed)) return HT_CLOSED;
    uint32_t snap=ld_u32(&q->tail), idx=snap&SLOT_MASK, want=gen_of(snap)*2u;
    if(ld_u32(&q->ids[idx])!=want) return HT_BUSY;
    if(atomicCAS((unsigned int*)&q->tail,snap,snap+1u)!=snap) return HT_BUSY;
    q->data[idx]=item; full_fence();
    atomicExch((unsigned int*)&q->ids[idx],want+1u); return HT_SUCCESS;
}

__device__ int ht_dequeue_nb(HTQueue* q,uint32_t* out)
{
    if(ld_i32(&q->closed)) return HT_CLOSED;
    uint32_t snap=ld_u32(&q->head), idx=snap&SLOT_MASK, want=gen_of(snap)*2u+1u;
    if(ld_u32(&q->ids[idx])!=want) return HT_BUSY;
    if(atomicCAS((unsigned int*)&q->head,snap,snap+1u)!=snap) return HT_BUSY;
    *out=ld_u32(&q->data[idx]); full_fence();
    atomicExch((unsigned int*)&q->ids[idx],want+1u); return HT_SUCCESS;
}

__global__ void mixedKernel(HTQueue* q,int* enqOK,int* deqOK,
                             uint32_t* buf,bool block,int iters)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=0;i<iters;++i){
        if(tid&1){
            uint32_t v; int st=block?ht_dequeue(q,&v):ht_dequeue_nb(q,&v);
            if(st==HT_SUCCESS){int pos=atomicAdd(deqOK,1); if(pos<QUEUE_SIZE) buf[pos]=v;}
        }else{
            uint32_t v=tid*1000+i; int st=block?ht_enqueue(q,v):ht_enqueue_nb(q,v);
            if(st==HT_SUCCESS) atomicAdd(enqOK,1);
        }
    }
}

__global__ void producerKernel(HTQueue* q,int N,int* enqOK)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return;
    if(ht_enqueue(q,1000+tid)==HT_SUCCESS) atomicAdd(enqOK,1);
}

__global__ void consumerKernel(HTQueue* q,int N,int* deqOK,uint32_t* buf)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return;
    uint32_t v; if(ht_dequeue(q,&v)==HT_SUCCESS){
        int pos=atomicAdd(deqOK,1); if(pos<N) buf[pos]=v;
    }
}

// -------------------------------
// New Testing Kernel for Timing
// -------------------------------
__global__ void testQueueOps(HTQueue* q, int opsPerThread, curandState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];

    for (int i = 0; i < opsPerThread; ++i) {
        int op = curand(&localState) % 2; // random enqueue or dequeue

        if (op == 0) {
            // Try non-blocking enqueue
            uint32_t val = tid * 10000 + i;
            for (int attempt = 0; attempt < 10; ++attempt) {
                if (ht_enqueue_nb(q, val) == HT_SUCCESS) break;
                tiny_delay();
            }
        } else {
            // Try non-blocking dequeue
            uint32_t out;
            for (int attempt = 0; attempt < 10; ++attempt) {
                if (ht_dequeue_nb(q, &out) == HT_SUCCESS) break;
                tiny_delay();
            }
        }
    }
    states[tid] = localState;
}

__global__ void setupRNG(curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// -------------------------------
// New Main Function
// -------------------------------
int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Blocks\tTime (ms)\n");

    for (int numBlocks = 1; numBlocks <= 49; ++numBlocks) {
        HTQueue *d_q;
        cudaMalloc(&d_q,sizeof(HTQueue));
        HTQueue h_q;
        initQueue(&h_q);
        cudaMemcpy(d_q, &h_q, sizeof(HTQueue), cudaMemcpyHostToDevice);

        curandState *d_states;
        cudaMalloc(&d_states, numBlocks * THREADS_PER_BLK * sizeof(curandState));
        setupRNG<<<numBlocks, THREADS_PER_BLK>>>(d_states, 1234);

        cudaDeviceSynchronize();

        cudaEventRecord(start);
        testQueueOps<<<numBlocks, THREADS_PER_BLK>>>(d_q, 1000 / (numBlocks * THREADS_PER_BLK) + 1, d_states);
        cudaEventRecord(stop);

        cudaDeviceSynchronize();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%d\t%.3f\n", numBlocks, milliseconds);

        cudaFree(d_q);
        cudaFree(d_states);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
}

