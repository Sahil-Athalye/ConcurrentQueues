#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>

// Queue size and maximum number of threads
#define QUEUE_SIZE 1024
#define MAX_THREADS 1024

// Return codes
#define SUCCESS 0
#define EMPTY 1
#define FULL 2
#define CLOSED 3

// Operations
#define OP_NONE 0
#define OP_ENQUEUE 1
#define OP_DEQUEUE 2

// Publication record status
#define STATUS_WAITING 0
#define STATUS_DONE 1

// Structure to represent an operation request
struct PublicationRecord {
    volatile int operation;     // OP_NONE, OP_ENQUEUE, OP_DEQUEUE
    volatile int status;        // STATUS_WAITING, STATUS_DONE
    volatile int value;         // Value for enqueue or dequeued value
    volatile int result;        // Result code
};

// Simple queue implementation used by the combiner
struct SimpleQueue {
    volatile int items[QUEUE_SIZE];
    volatile int head;
    volatile int tail;
    volatile int count;
};

// The flat-combining queue structure
struct FCQueue {
    SimpleQueue queue;
    volatile int closed;
    volatile int lock;
    volatile int combinerActive;
    PublicationRecord publications[MAX_THREADS];
    
    // Statistics
    volatile int numCombines;
    volatile int totalOpsProcessed;
};

// Initialize the FC queue
__host__ void initFCQueue(FCQueue* queue) {
    queue->queue.head = 0;
    queue->queue.tail = 0;
    queue->queue.count = 0;
    queue->closed = 0;
    queue->lock = 0;
    queue->combinerActive = 0;
    queue->numCombines = 0;
    queue->totalOpsProcessed = 0;
    
    // Initialize all publication records
    for (int i = 0; i < MAX_THREADS; i++) {
        queue->publications[i].operation = OP_NONE;
        queue->publications[i].status = STATUS_DONE;
        queue->publications[i].value = 0;
        queue->publications[i].result = SUCCESS;
    }
}

// Helper function to try to acquire the lock
__device__ bool tryLock(volatile int* lock) {
    return (atomicCAS((int*)lock, 0, 1) == 0);
}

// Helper function to release the lock
__device__ void unlock(volatile int* lock) {
    atomicExch((int*)lock, 0);
}

// Simple queue operations used by the combiner
__device__ bool simpleEnqueue(SimpleQueue* queue, int value) {
    if (queue->count >= QUEUE_SIZE) {
        return false;  // Queue is full
    }
    
    queue->items[queue->tail] = value;
    queue->tail = (queue->tail + 1) % QUEUE_SIZE;
    queue->count++;
    return true;
}

__device__ bool simpleDequeue(SimpleQueue* queue, int* value) {
    if (queue->count <= 0) {
        return false;  // Queue is empty
    }
    
    *value = queue->items[queue->head];
    queue->head = (queue->head + 1) % QUEUE_SIZE;
    queue->count--;
    return true;
}

// Become the combiner and process all pending operations
__device__ void doCombining(FCQueue* fcq) {
    atomicExch((int*)&fcq->combinerActive, 1);
    int opsProcessed = 0;
    
    for (int i = 0; i < MAX_THREADS; i++) {
        PublicationRecord* pub = &fcq->publications[i];
        
        if (pub->operation != OP_NONE && pub->status == STATUS_WAITING) {
            if (pub->operation == OP_ENQUEUE) {
                bool success = simpleEnqueue(&fcq->queue, pub->value);
                pub->result = success ? SUCCESS : FULL;
            }
            else if (pub->operation == OP_DEQUEUE) {
                int value;
                bool success = simpleDequeue(&fcq->queue, &value);
                if (success) {
                    pub->value = value;
                    pub->result = SUCCESS;
                } else {
                    pub->result = EMPTY;
                }
            }
            
            atomicExch((int*)&pub->status, STATUS_DONE);
            opsProcessed++;
        }
    }
    
    atomicAdd((int*)&fcq->numCombines, 1);
    atomicAdd((int*)&fcq->totalOpsProcessed, opsProcessed);
    
    atomicExch((int*)&fcq->combinerActive, 0);
    unlock(&fcq->lock);
}

// Publish an operation and wait for it to be processed
__device__ int publishOperation(FCQueue* fcq, int threadId, int operation, int value) {
    if (atomicAdd((int*)&fcq->closed, 0) != 0) {
        return CLOSED;
    }
    
    PublicationRecord* pub = &fcq->publications[threadId];
    pub->operation = operation;
    pub->value = value;
    pub->result = SUCCESS;
    
    atomicExch((int*)&pub->status, STATUS_WAITING);
    __threadfence();
    
    if (atomicAdd((int*)&fcq->combinerActive, 0) == 0 && tryLock(&fcq->lock)) {
        doCombining(fcq);
    }
    
    while (atomicAdd((int*)&pub->status, 0) == STATUS_WAITING) {
        if (atomicAdd((int*)&fcq->combinerActive, 0) == 0 && tryLock(&fcq->lock)) {
            doCombining(fcq);
        }
        
        for (int i = 0; i < 32; i++) {
            __threadfence();
        }
        
        if (atomicAdd((int*)&fcq->closed, 0) != 0) {
            if (atomicCAS((int*)&pub->status, STATUS_WAITING, STATUS_DONE) == STATUS_WAITING) {
                pub->operation = OP_NONE;
                return CLOSED;
            }
        }
    }
    
    int result = pub->result;
    pub->operation = OP_NONE;
    
    return result;
}

// Enqueue an item into the FC queue
__device__ int enqueue(FCQueue* fcq, int threadId, int value) {
    return publishOperation(fcq, threadId, OP_ENQUEUE, value);
}

// Dequeue an item from the FC queue
__device__ int dequeue(FCQueue* fcq, int threadId, int* value) {
    int result = publishOperation(fcq, threadId, OP_DEQUEUE, 0);
    if (result == SUCCESS) {
        *value = fcq->publications[threadId].value;
    }
    return result;
}

// CUDA kernel to test the FC queue with mixed operations
__global__ void testFCQueueKernel(FCQueue* fcq, int* enqueueSuccess, int* dequeueSuccess, 
                                   int* results, int numIterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < MAX_THREADS) {
        for (int i = 0; i < numIterations; i++) {
            if ((tid % 2 == 0) || (tid % 7 == 0)) {
                int result = enqueue(fcq, tid, tid * 1000 + i);
                if (result == SUCCESS) {
                    atomicAdd(enqueueSuccess, 1);
                }
            } else {
                int value;
                int result = dequeue(fcq, tid, &value);
                if (result == SUCCESS) {
                    int idx = atomicAdd(dequeueSuccess, 1);
                    if (idx < QUEUE_SIZE) {
                        results[idx] = value;
                    }
                }
            }
        }
    }
}

// Main function for testing the FC queue with timing
int main() {
    FCQueue* d_fcq;
    cudaMalloc(&d_fcq, sizeof(FCQueue));
    
    FCQueue h_fcq;
    initFCQueue(&h_fcq);
    
    cudaMemcpy(d_fcq, &h_fcq, sizeof(FCQueue), cudaMemcpyHostToDevice);
    
    int* d_enqueueSuccess;
    int* d_dequeueSuccess;
    int* d_results;
    int* d_numCombines;
    int* d_totalOpsProcessed;
    
    cudaMalloc(&d_enqueueSuccess, sizeof(int));
    cudaMalloc(&d_dequeueSuccess, sizeof(int));
    cudaMalloc(&d_results, QUEUE_SIZE * sizeof(int));
    cudaMalloc(&d_numCombines, sizeof(int));
    cudaMalloc(&d_totalOpsProcessed, sizeof(int));
    
    int zero = 0;
    cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Test general mixed operations
    int numIterations = 1000;
    
    for (int numBlocks = 1; numBlocks <= 49; numBlocks++) {
        int totalThreads = 32 * numBlocks;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        testFCQueueKernel<<<numBlocks, 32>>>(d_fcq, d_enqueueSuccess, d_dequeueSuccess, d_results, numIterations);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Configuration: %d blocks, %d threads per block\n", numBlocks, 32);
        printf("Time taken: %f ms\n", milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_results);
    cudaFree(d_enqueueSuccess);
    cudaFree(d_dequeueSuccess);
    cudaFree(d_numCombines);
    cudaFree(d_totalOpsProcessed);
    cudaFree(d_fcq);
    
    return 0;
}
