/**********************************************************************
 *  High-Throughput Blocking Array Queue – 32-bit safe (sm_60+)
 *  nvcc -std=c++17 -arch=sm_60 ht_blocking_queue.cu
 *********************************************************************/
 #include <cuda_runtime.h>
 #include <stdint.h>
 #include <stdio.h>
 #include <assert.h>
 
 // ------------------------------------------------------------------
 //  CONSTANTS
 // ------------------------------------------------------------------
 #define QUEUE_SIZE     1024
 #define LG_QSIZE       10                    // log2(1024)
 #define SLOT_MASK      (QUEUE_SIZE-1)
 #define THREADS_PER_BLK 128
 #define BLOCKS         10
 
 static_assert((QUEUE_SIZE&(QUEUE_SIZE-1))==0,"size must be power-of-two");
 
 // ------------------------------------------------------------------
 //  INLINE HELPERS
 // ------------------------------------------------------------------
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
 
 // ------------------------------------------------------------------
 //  QUEUE STRUCTURE
 // ------------------------------------------------------------------
 struct HTQueue {
     volatile uint32_t head, tail;
     volatile int32_t  closed;
     volatile uint32_t data[QUEUE_SIZE];
     volatile uint32_t ids [QUEUE_SIZE];   // generation*2 | state
 };
 
 enum { HT_SUCCESS=0, HT_CLOSED=1, HT_BUSY=2 };
 
 // ------------------------------------------------------------------
 //  HOST INITIALISER
 // ------------------------------------------------------------------
 __host__ void initQueue(HTQueue* q)
 {
     q->head = q->tail = 0;  q->closed = 0;
     for (int i=0;i<QUEUE_SIZE;++i) q->ids[i]=0;
 }
 
 // ------------------------------------------------------------------
 //  DEVICE  –  BLOCKING OPS (32-bit atomics only)
 // ------------------------------------------------------------------
 __device__ int ht_enqueue(HTQueue* q,uint32_t item)
 {
     for(;;){
         if (ld_i32(&q->closed)) return HT_CLOSED;
 
         uint32_t snap = ld_u32(&q->tail);
         uint32_t idx  = snap & SLOT_MASK;
         uint32_t want = gen_of(snap)*2u;          // even → free
 
         if (ld_u32(&q->ids[idx]) != want){ tiny_delay(); continue;}
 
         if (atomicCAS((unsigned int*)&q->tail,snap,snap+1u)!=snap){
             tiny_delay(); continue;
         }
         q->data[idx]=item; full_fence();
         atomicExch((unsigned int*)&q->ids[idx], want+1u); // odd
         return HT_SUCCESS;
     }
 }
 
 __device__ int ht_dequeue(HTQueue* q,uint32_t* out)
 {
     for(;;){
         if (ld_i32(&q->closed)) return HT_CLOSED;
 
         uint32_t snap = ld_u32(&q->head);
         uint32_t idx  = snap & SLOT_MASK;
         uint32_t want = gen_of(snap)*2u+1u;       // odd → full
 
         if (ld_u32(&q->ids[idx]) != want){ tiny_delay(); continue;}
 
         if (atomicCAS((unsigned int*)&q->head,snap,snap+1u)!=snap){
             tiny_delay(); continue;
         }
         *out = ld_u32(&q->data[idx]); full_fence();
         atomicExch((unsigned int*)&q->ids[idx], want+1u); // next even
         return HT_SUCCESS;
     }
 }
 
 // ------------------------------------------------------------------
 //  NON-BLOCKING (unchanged but 32-bit)
 // ------------------------------------------------------------------
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
 
 // ------------------------------------------------------------------
 //  TEST KERNELS – unchanged logic
 // ------------------------------------------------------------------
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
 
 // ------------------------------------------------------------------
 //  HOST DRIVER
 // ------------------------------------------------------------------
 int main()
 {
     HTQueue *d_q; cudaMalloc(&d_q,sizeof(HTQueue));
     HTQueue  h_q; initQueue(&h_q);
     cudaMemcpy(d_q,&h_q,sizeof(HTQueue),cudaMemcpyHostToDevice);
 
     int *d_enq,*d_deq; cudaMalloc(&d_enq,sizeof(int));
     cudaMalloc(&d_deq,sizeof(int));
     uint32_t *d_buf; cudaMalloc(&d_buf,QUEUE_SIZE*sizeof(uint32_t));
     int zero=0; cudaMemcpy(d_enq,&zero,4,cudaMemcpyHostToDevice);
                cudaMemcpy(d_deq,&zero,4,cudaMemcpyHostToDevice);
 
     // Blocking mixed
     printf("Testing *blocking* operations ...\n");
     mixedKernel<<<BLOCKS,THREADS_PER_BLK>>>(d_q,d_enq,d_deq,d_buf,true,5);
     cudaDeviceSynchronize();
     int enqOK,deqOK; cudaMemcpy(&enqOK,d_enq,4,cudaMemcpyDeviceToHost);
     cudaMemcpy(&deqOK,d_deq,4,cudaMemcpyDeviceToHost);
     printf("  Enqueue successes: %d\n",enqOK);
     printf("  Dequeue successes: %d\n",deqOK);
 
     // Non-blocking mixed
     cudaMemcpy(d_q,&h_q,sizeof(HTQueue),cudaMemcpyHostToDevice);
     cudaMemcpy(d_enq,&zero,4,cudaMemcpyHostToDevice);
     cudaMemcpy(d_deq,&zero,4,cudaMemcpyHostToDevice);
     printf("\nTesting *non-blocking* operations ...\n");
     mixedKernel<<<BLOCKS,THREADS_PER_BLK>>>(d_q,d_enq,d_deq,d_buf,false,5);
     cudaDeviceSynchronize();
     cudaMemcpy(&enqOK,d_enq,4,cudaMemcpyDeviceToHost);
     cudaMemcpy(&deqOK,d_deq,4,cudaMemcpyDeviceToHost);
     printf("  Enqueue successes: %d\n",enqOK);
     printf("  Dequeue successes: %d\n",deqOK);
 
     // Producer / consumer
     cudaMemcpy(d_q,&h_q,sizeof(HTQueue),cudaMemcpyHostToDevice);
     cudaMemcpy(d_enq,&zero,4,cudaMemcpyHostToDevice);
     cudaMemcpy(d_deq,&zero,4,cudaMemcpyHostToDevice);
     const int N=500;
     printf("\nTesting producer / consumer pattern ...\n");
     producerKernel<<<(N+THREADS_PER_BLK-1)/THREADS_PER_BLK,
                      THREADS_PER_BLK>>>(d_q,N,d_enq);
     consumerKernel<<<(N+THREADS_PER_BLK-1)/THREADS_PER_BLK,
                      THREADS_PER_BLK>>>(d_q,N,d_deq,d_buf);
     cudaDeviceSynchronize();
     cudaMemcpy(&enqOK,d_enq,4,cudaMemcpyDeviceToHost);
     cudaMemcpy(&deqOK,d_deq,4,cudaMemcpyDeviceToHost);
     printf("  Enqueue successes: %d\n",enqOK);
     printf("  Dequeue successes: %d\n",deqOK);
 
     cudaFree(d_buf); cudaFree(d_enq); cudaFree(d_deq); cudaFree(d_q);
     cudaDeviceReset();
     return 0;
 }
 