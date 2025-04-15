/**
 * Flat-Combining Queue Implementation in CUDA
 * Based on the concept from "Flat Combining and the Synchronization-Parallelism Tradeoff"
 * 
 * This implementation provides a combining-based concurrent queue for CUDA.
 * Instead of having each thread perform its own operation, threads publish their operations
 * and a single combiner thread performs all operations in a batch.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 
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
 
 /**
  * Structure to represent an operation request
  */
 struct PublicationRecord {
     volatile int operation;     // OP_NONE, OP_ENQUEUE, OP_DEQUEUE
     volatile int status;        // STATUS_WAITING, STATUS_DONE
     volatile int value;         // Value for enqueue or dequeued value
     volatile int result;        // Result code
 };
 
 /**
  * Simple queue implementation used by the combiner
  */
 struct SimpleQueue {
     volatile int items[QUEUE_SIZE];
     volatile int head;
     volatile int tail;
     volatile int count;
 };
 
 /**
  * The flat-combining queue structure
  */
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
 
 /**
  * Initialize the FC queue
  */
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
 
 /**
  * Helper function to try to acquire the lock
  */
 __device__ bool tryLock(volatile int* lock) {
     return (atomicCAS((int*)lock, 0, 1) == 0);
 }
 
 /**
  * Helper function to release the lock
  */
 __device__ void unlock(volatile int* lock) {
     atomicExch((int*)lock, 0);
 }
 
 /**
  * Simple queue operations used by the combiner
  */
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
 
 /**
  * Become the combiner and process all pending operations
  */
 __device__ void doCombining(FCQueue* fcq) {
     // Set the combiner flag
     atomicExch((int*)&fcq->combinerActive, 1);
     int opsProcessed = 0;
     
     // Scan through all publication records
     for (int i = 0; i < MAX_THREADS; i++) {
         PublicationRecord* pub = &fcq->publications[i];
         
         // Check if this record has a pending operation
         if (pub->operation != OP_NONE && pub->status == STATUS_WAITING) {
             if (pub->operation == OP_ENQUEUE) {
                 // Process enqueue request
                 bool success = simpleEnqueue(&fcq->queue, pub->value);
                 if (success) {
                     pub->result = SUCCESS;
                 } else {
                     pub->result = FULL;
                 }
             }
             else if (pub->operation == OP_DEQUEUE) {
                 // Process dequeue request
                 int value;
                 bool success = simpleDequeue(&fcq->queue, &value);
                 if (success) {
                     pub->value = value;
                     pub->result = SUCCESS;
                 } else {
                     pub->result = EMPTY;
                 }
             }
             
             // Mark the operation as completed
             atomicExch((int*)&pub->status, STATUS_DONE);
             opsProcessed++;
         }
     }
     
     // Update statistics
     atomicAdd((int*)&fcq->numCombines, 1);
     atomicAdd((int*)&fcq->totalOpsProcessed, opsProcessed);
     
     // Release the combiner flag and unlock
     atomicExch((int*)&fcq->combinerActive, 0);
     unlock(&fcq->lock);
 }
 
 /**
  * Publish an operation and wait for it to be processed
  */
 __device__ int publishOperation(FCQueue* fcq, int threadId, int operation, int value) {
     if (atomicAdd((int*)&fcq->closed, 0) != 0) {
         return CLOSED;
     }
     
     // Get the publication record for this thread
     PublicationRecord* pub = &fcq->publications[threadId];
     
     // Fill in the operation details
     pub->operation = operation;
     pub->value = value;
     pub->result = SUCCESS;
     
     // Mark as waiting and memory fence to ensure visibility
     atomicExch((int*)&pub->status, STATUS_WAITING);
     __threadfence();
     
     // Try to become the combiner if no one is active
     if (atomicAdd((int*)&fcq->combinerActive, 0) == 0 && tryLock(&fcq->lock)) {
         doCombining(fcq);
     }
     
     // Wait for the operation to complete
     while (atomicAdd((int*)&pub->status, 0) == STATUS_WAITING) {
         // If no combiner is active, try to become one
         if (atomicAdd((int*)&fcq->combinerActive, 0) == 0 && tryLock(&fcq->lock)) {
             doCombining(fcq);
         }
         
         // Simple backoff
         for (int i = 0; i < 32; i++) {
             __threadfence();
         }
         
         // Check if the queue was closed while waiting
         if (atomicAdd((int*)&fcq->closed, 0) != 0) {
             // Try to reset our publication if still waiting
             if (atomicCAS((int*)&pub->status, STATUS_WAITING, STATUS_DONE) == STATUS_WAITING) {
                 pub->operation = OP_NONE;
                 return CLOSED;
             }
         }
     }
     
     // Get the result
     int result = pub->result;
     
     // Reset the publication record
     pub->operation = OP_NONE;
     
     return result;
 }
 
 /**
  * Enqueue an item into the FC queue
  */
 __device__ int enqueue(FCQueue* fcq, int threadId, int value) {
     return publishOperation(fcq, threadId, OP_ENQUEUE, value);
 }
 
 /**
  * Dequeue an item from the FC queue
  */
 __device__ int dequeue(FCQueue* fcq, int threadId, int* value) {
     int result = publishOperation(fcq, threadId, OP_DEQUEUE, 0);
     if (result == SUCCESS) {
         *value = fcq->publications[threadId].value;
     }
     return result;
 }
 
 /**
  * Close the FC queue
  */
 __device__ void closeQueue(FCQueue* fcq) {
     atomicExch((int*)&fcq->closed, 1);
 }
 
 /**
  * Get the number of items in the queue
  */
 __device__ int getQueueSize(FCQueue* fcq) {
     // Need to acquire the lock to get an accurate count
     if (tryLock(&fcq->lock)) {
         int count = fcq->queue.count;
         unlock(&fcq->lock);
         return count;
     }
     // If can't acquire lock, return an approximate count
     return fcq->queue.count;
 }
 
 /**
  * CUDA kernel to test the FC queue with mixed operations
  */
 __global__ void testFCQueueKernel(FCQueue* fcq, int* enqueueSuccess, int* dequeueSuccess, 
                                    int* results, int numIterations) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     if (tid < MAX_THREADS) {
         for (int i = 0; i < numIterations; i++) {
             // Even threads mostly enqueue, odd threads mostly dequeue
             if ((tid % 2 == 0) || (tid % 7 == 0)) {  // Add some variation
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
 
 /**
  * CUDA kernel for producer threads (only enqueue)
  */
 __global__ void producerKernel(FCQueue* fcq, int* enqueueSuccess, int startValue, int numItems) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     int localThreadId = tid % MAX_THREADS; // Ensure thread ID is within bounds
     
     int itemsPerThread = (numItems + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
     int startItem = tid * itemsPerThread;
     int endItem = min(startItem + itemsPerThread, numItems);
     
     for (int i = startItem; i < endItem; i++) {
         int result = enqueue(fcq, localThreadId, startValue + i);
         if (result == SUCCESS) {
             atomicAdd(enqueueSuccess, 1);
         }
     }
 }
 
 /**
  * CUDA kernel for consumer threads (only dequeue)
  */
 __global__ void consumerKernel(FCQueue* fcq, int* dequeueSuccess, int* results, int numItems) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     int localThreadId = tid % MAX_THREADS; // Ensure thread ID is within bounds
     
     int itemsPerThread = (numItems + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
     int startItem = tid * itemsPerThread;
     int endItem = min(startItem + itemsPerThread, numItems);
     
     for (int i = startItem; i < endItem; i++) {
         int value;
         int result = dequeue(fcq, localThreadId, &value);
         if (result == SUCCESS) {
             int idx = atomicAdd(dequeueSuccess, 1);
             if (idx < numItems) {
                 results[idx] = value;
             }
         }
     }
 }
 
 /**
  * CUDA kernel to get statistics
  */
 __global__ void getStatsKernel(FCQueue* fcq, int* numCombines, int* totalOpsProcessed) {
     *numCombines = fcq->numCombines;
     *totalOpsProcessed = fcq->totalOpsProcessed;
 }
 
 /**
  * Main function to test the FC Queue
  */
 int main() {
     // Allocate FC queue on host and device
     FCQueue* d_fcq;
     cudaMalloc(&d_fcq, sizeof(FCQueue));
     
     // Initialize queue on the host
     FCQueue h_fcq;
     initFCQueue(&h_fcq);
     
     // Copy to device
     cudaMemcpy(d_fcq, &h_fcq, sizeof(FCQueue), cudaMemcpyHostToDevice);
     
     // Allocate memory for results and counters
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
     
     // Initialize counters
     int zero = 0;
     cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Test general mixed operations
     printf("Testing FC queue with mixed operations...\n");
     int numIterations = 10;
     testFCQueueKernel<<<10, 100>>>(d_fcq, d_enqueueSuccess, d_dequeueSuccess, d_results, numIterations);
     cudaDeviceSynchronize();
     
     // Copy results back
     int h_enqueueSuccess, h_dequeueSuccess;
     int h_numCombines, h_totalOpsProcessed;
     
     cudaMemcpy(&h_enqueueSuccess, d_enqueueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeueSuccess, d_dequeueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     
     // Get statistics
     getStatsKernel<<<1, 1>>>(d_fcq, d_numCombines, d_totalOpsProcessed);
     cudaMemcpy(&h_numCombines, d_numCombines, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_totalOpsProcessed, d_totalOpsProcessed, sizeof(int), cudaMemcpyDeviceToHost);
     
     printf("Mixed operations test results:\n");
     printf("  Enqueue successes: %d\n", h_enqueueSuccess);
     printf("  Dequeue successes: %d\n", h_dequeueSuccess);
     printf("  Number of combines: %d\n", h_numCombines);
     printf("  Total operations processed: %d\n", h_totalOpsProcessed);
     printf("  Average operations per combine: %.2f\n", 
            h_numCombines > 0 ? (float)h_totalOpsProcessed / h_numCombines : 0);
     
     // Reset counters for producer-consumer test
     cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_numCombines, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_totalOpsProcessed, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Reset queue
     initFCQueue(&h_fcq);
     cudaMemcpy(d_fcq, &h_fcq, sizeof(FCQueue), cudaMemcpyHostToDevice);
     
     // Test producer-consumer pattern
     printf("\nTesting FC queue with producer-consumer pattern...\n");
     int numItems = 500;
     
     // Launch producer kernel
     producerKernel<<<5, 50>>>(d_fcq, d_enqueueSuccess, 1000, numItems);
     cudaDeviceSynchronize();
     
     // Launch consumer kernel
     consumerKernel<<<5, 50>>>(d_fcq, d_dequeueSuccess, d_results, numItems);
     cudaDeviceSynchronize();
     
     // Copy results back
     cudaMemcpy(&h_enqueueSuccess, d_enqueueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeueSuccess, d_dequeueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     
     // Get statistics
     getStatsKernel<<<1, 1>>>(d_fcq, d_numCombines, d_totalOpsProcessed);
     cudaMemcpy(&h_numCombines, d_numCombines, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_totalOpsProcessed, d_totalOpsProcessed, sizeof(int), cudaMemcpyDeviceToHost);
     
     printf("Producer-Consumer test results:\n");
     printf("  Enqueue successes: %d\n", h_enqueueSuccess);
     printf("  Dequeue successes: %d\n", h_dequeueSuccess);
     printf("  Number of combines: %d\n", h_numCombines);
     printf("  Total operations processed: %d\n", h_totalOpsProcessed);
     printf("  Average operations per combine: %.2f\n", 
            h_numCombines > 0 ? (float)h_totalOpsProcessed / h_numCombines : 0);
     
     // Cleanup
     cudaFree(d_results);
     cudaFree(d_enqueueSuccess);
     cudaFree(d_dequeueSuccess);
     cudaFree(d_numCombines);
     cudaFree(d_totalOpsProcessed);
     cudaFree(d_fcq);
     
     return 0;
 }