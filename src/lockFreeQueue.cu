/**
 * True Concurrent Michael-Scott Lock-Free Queue Implementation in CUDA
 * 
 * This version allows for simultaneous enqueue and dequeue operations
 * with proper handling of contention and memory ordering.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 
 // Constants for the queue
 #define QUEUE_SIZE 1024
 #define MAX_THREADS 1000
 #define MAX_RETRIES 10
 
 /**
  * Node structure with value and next pointer
  */
 struct Node {
     int value;
     volatile int next;       // Index to the next node in the pool (or -1 for NULL)
     volatile int nextCount;  // Counter to prevent ABA problem
 };
 
 /**
  * Queue structure with head and tail indices
  */
 struct MSQueue {
     volatile int head;        // Index to the head node
     volatile int headCount;   // Counter for head
     volatile int tail;        // Index to the tail node
     volatile int tailCount;   // Counter for tail
     Node nodes[QUEUE_SIZE];   // Pre-allocated pool of nodes
     volatile int freeList;    // Index to the first free node
     volatile int freeCount;   // Number of nodes in the free list
 };
 
 // Error codes
 #define SUCCESS 0
 #define EMPTY 1
 #define FULL 2
 
 /**
  * Initialize the queue on the host
  */
 __host__ void initQueue(MSQueue* queue) {
     // Initialize all nodes in the pool
     for (int i = 0; i < QUEUE_SIZE - 1; i++) {
         queue->nodes[i].next = i + 1;
         queue->nodes[i].nextCount = 0;
     }
     
     // Last node points to nothing
     queue->nodes[QUEUE_SIZE - 1].next = -1;
     queue->nodes[QUEUE_SIZE - 1].nextCount = 0;
     
     // Initialize free list
     queue->freeList = 1;  // Reserve node 0 as the initial dummy node
     queue->freeCount = QUEUE_SIZE - 1;
     
     // Set up the dummy node
     queue->nodes[0].value = -1;  // Invalid value
     queue->nodes[0].next = -1;   // No next node initially
     queue->nodes[0].nextCount = 0;
     
     // Head and tail both point to the dummy node
     queue->head = 0;
     queue->headCount = 0;
     queue->tail = 0;
     queue->tailCount = 0;
     
     printf("Queue initialized with dummy node at index 0\n");
     printf("Free list starts at index %d with %d nodes\n", queue->freeList, queue->freeCount);
 }
 
 /**
  * Get a free node from the pool
  * 
  * @param queue The queue
  * @return Index of the allocated node, or -1 if no free nodes
  */
 __device__ int allocateNode(MSQueue* queue) {
     int retries = 0;
     while (retries++ < MAX_RETRIES) {
         // Check if we have free nodes
         int freeList = atomicAdd((int*)&queue->freeList, 0);
         if (freeList == -1) {
             return -1;  // No free nodes
         }
         
         // Try to take the first free node
         int nextFree = queue->nodes[freeList].next;
         if (atomicCAS((int*)&queue->freeList, freeList, nextFree) == freeList) {
             atomicSub((int*)&queue->freeCount, 1);
             return freeList;
         }
         
         // If CAS failed, another thread took the node, try again
     }
     return -1; // Too many retries, give up
 }
 
 /**
  * Return a node to the free list
  * 
  * @param queue The queue
  * @param nodeIndex Index of the node to free
  */
 __device__ void freeNode(MSQueue* queue, int nodeIndex) {
     int retries = 0;
     while (retries++ < MAX_RETRIES) {
         int currentFreeList = atomicAdd((int*)&queue->freeList, 0);
         queue->nodes[nodeIndex].next = currentFreeList;
         if (atomicCAS((int*)&queue->freeList, currentFreeList, nodeIndex) == currentFreeList) {
             atomicAdd((int*)&queue->freeCount, 1);
             return;
         }
     }
     // If we couldn't free after several retries, something is wrong but we don't want to block
 }
 
 /**
  * Enqueue an item to the queue
  * 
  * @param queue The queue to enqueue to
  * @param value The value to enqueue
  * @return 0 on success, FULL if queue is full or too many retries
  */
 __device__ int enqueue(MSQueue* queue, int value) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     // Get a new node from the pool
     int nodeIndex = allocateNode(queue);
     if (nodeIndex == -1) {
         return FULL;
     }
     
     // Initialize the new node
     queue->nodes[nodeIndex].value = value;
     queue->nodes[nodeIndex].next = -1;
     __threadfence();  // Ensure the node initialization is visible to other threads
     
     int retries = 0;
     while (retries++ < MAX_RETRIES) {
         // Read the current tail
         int tail = atomicAdd((int*)&queue->tail, 0);
         int tailCount = atomicAdd((int*)&queue->tailCount, 0);
         
         // Read the next pointer of the tail node
         int next = atomicAdd((int*)&queue->nodes[tail].next, 0);
         int nextCount = atomicAdd((int*)&queue->nodes[tail].nextCount, 0);
         
         // Check if tail is still consistent
         if (tail != atomicAdd((int*)&queue->tail, 0) || 
             tailCount != atomicAdd((int*)&queue->tailCount, 0)) {
             continue;
         }
         
         if (next == -1) {
             // Try to link the new node
             if (atomicCAS((int*)&queue->nodes[tail].next, -1, nodeIndex) == -1) {
                 // Successfully linked, increment the nextCount
                 atomicAdd((int*)&queue->nodes[tail].nextCount, 1);
                 __threadfence();  // Ensure the link is visible before advancing tail
                 
                 // Now try to advance the tail
                 atomicCAS((int*)&queue->tail, tail, nodeIndex);
                 atomicAdd((int*)&queue->tailCount, 1);
                 
                 return SUCCESS;
             }
         } else {
             // Tail is not pointing to the last node, try to advance it
             atomicCAS((int*)&queue->tail, tail, next);
             atomicAdd((int*)&queue->tailCount, 1);
         }
     }
     
     // If we failed after max retries, free the node and return
     freeNode(queue, nodeIndex);
     return FULL;
 }
 
 /**
  * Dequeue an item from the queue
  * 
  * @param queue The queue to dequeue from
  * @param value Pointer to store the dequeued value
  * @return 0 on success, EMPTY if queue was empty or too many retries
  */
 __device__ int dequeue(MSQueue* queue, int* value) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     int retries = 0;
     while (retries++ < MAX_RETRIES) {
         // Read the current head and tail
         int head = atomicAdd((int*)&queue->head, 0);
         int headCount = atomicAdd((int*)&queue->headCount, 0);
         int tail = atomicAdd((int*)&queue->tail, 0);
         
         // Read the next pointer of the head node
         int next = atomicAdd((int*)&queue->nodes[head].next, 0);
         
         // Check if head is still consistent
         if (head != atomicAdd((int*)&queue->head, 0) || 
             headCount != atomicAdd((int*)&queue->headCount, 0)) {
             continue;
         }
         
         if (head == tail) {
             // Queue might be empty
             if (next == -1) {
                 return EMPTY;  // Queue is definitely empty
             }
             
             // Tail is falling behind, try to advance it
             atomicCAS((int*)&queue->tail, tail, next);
             atomicAdd((int*)&queue->tailCount, 1);
         } else {
             // Process dequeue
             if (next == -1) {
                 // This shouldn't happen in normal operation
                 continue;
             }
             
             // Read the value from the next node (real head of the queue)
             *value = queue->nodes[next].value;
             
             // Try to advance head
             if (atomicCAS((int*)&queue->head, head, next) == head) {
                 atomicAdd((int*)&queue->headCount, 1);
                 __threadfence();  // Ensure head update is visible
                 
                 // Free the old dummy node - can be done asynchronously
                 freeNode(queue, head);
                 return SUCCESS;
             }
         }
     }
     
     // If we failed after max retries, return empty
     return EMPTY;
 }
 
 /**
  * Print the current state of the queue (for debugging)
  */
 __global__ void printQueueState(MSQueue* queue) {
     // Only one thread should print
     if (threadIdx.x == 0 && blockIdx.x == 0) {
         printf("\nQueue State:\n");
         printf("Head index: %d, Head count: %d\n", 
                queue->head, queue->headCount);
         printf("Tail index: %d, Tail count: %d\n", 
                queue->tail, queue->tailCount);
         
         // Print first few nodes
         int node = queue->head;
         int count = 0;
         printf("Nodes starting from head:\n");
         while (node != -1 && count < 10) {
             printf("Node[%d]: value=%d, next=%d, nextCount=%d\n", 
                    node, queue->nodes[node].value, 
                    queue->nodes[node].next, queue->nodes[node].nextCount);
             node = queue->nodes[node].next;
             count++;
         }
         
         printf("Free list: %d nodes available\n", queue->freeCount);
     }
 }
 
 /**
  * Concurrent kernel with mixed enqueue and dequeue operations
  */
 __global__ void concurrentOperationsKernel(MSQueue* queue, int* results, int* enqueue_count, int* dequeue_count, int iterations) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     for (int i = 0; i < iterations; i++) {
         // Even threads enqueue, odd threads dequeue
         if (tid % 2 == 0) {
             int value = tid * 1000 + i; // Create unique values
             if (enqueue(queue, value) == SUCCESS) {
                 atomicAdd(enqueue_count, 1);
             }
         } else {
             int value;
             if (dequeue(queue, &value) == SUCCESS) {
                 int idx = atomicAdd(dequeue_count, 1);
                 if (idx < QUEUE_SIZE) {
                     results[idx] = value;
                 }
             }
         }
         
         // Small delay to vary timing between threads
         for (int j = 0; j < tid % 10; j++) {
             __threadfence_block();
         }
     }
 }
 
 /**
  * Main function to set up and run the test
  */
 int main() {
     // Allocate the queue on the device
     MSQueue* d_queue;
     cudaMalloc(&d_queue, sizeof(MSQueue));
     
     // Initialize the queue on the host
     MSQueue h_queue;
     initQueue(&h_queue);
     
     // Copy the initialized queue to the device
     cudaMemcpy(d_queue, &h_queue, sizeof(MSQueue), cudaMemcpyHostToDevice);
     
     // Allocate memory for results and counters
     int* d_results;
     int* d_enqueue_count;
     int* d_dequeue_count;
     cudaMalloc(&d_results, QUEUE_SIZE * sizeof(int));
     cudaMalloc(&d_enqueue_count, sizeof(int));
     cudaMalloc(&d_dequeue_count, sizeof(int));
     
     // Initialize counters
     int zero = 0;
     cudaMemcpy(d_enqueue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Set up the concurrent test
     int threadsPerBlock = 32;
     int numBlocks = 4;
     int iterations = 5; // Each thread performs this many operations
     int totalThreads = threadsPerBlock * numBlocks;
     
     printf("Starting concurrent test with %d threads, %d iterations each\n", 
            totalThreads, iterations);
     printf("Expected operations: ~%d enqueues and ~%d dequeues\n", 
            (totalThreads / 2) * iterations, (totalThreads / 2) * iterations);
     
     // Print initial queue state
     printf("\nInitial queue state:\n");
     printQueueState<<<1, 1>>>(d_queue);
     cudaDeviceSynchronize();
     
     // Launch concurrent operations kernel
     concurrentOperationsKernel<<<numBlocks, threadsPerBlock>>>(
         d_queue, d_results, d_enqueue_count, d_dequeue_count, iterations);
     cudaDeviceSynchronize();
     
     // Copy back the results
     int h_enqueue_count, h_dequeue_count;
     int* h_results = new int[QUEUE_SIZE];
     cudaMemcpy(&h_enqueue_count, d_enqueue_count, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeue_count, d_dequeue_count, sizeof(int), cudaMemcpyDeviceToHost);
     
     if (h_dequeue_count > 0) {
         cudaMemcpy(h_results, d_results, h_dequeue_count * sizeof(int), cudaMemcpyDeviceToHost);
     }
     
     // Print results
     printf("\nConcurrent test results:\n");
     printf("Enqueued: %d items\n", h_enqueue_count);
     printf("Dequeued: %d items\n", h_dequeue_count);
     
     if (h_dequeue_count > 0) {
         printf("First few dequeued values: ");
         int showCount = min(h_dequeue_count, 20);
         for (int i = 0; i < showCount; i++) {
             printf("%d ", h_results[i]);
         }
         if (h_dequeue_count > 20) {
             printf("... (showing first 20 only)");
         }
         printf("\n");
     }
     
     // Print final queue state
     printf("\nFinal queue state:\n");
     printQueueState<<<1, 1>>>(d_queue);
     cudaDeviceSynchronize();
     
     // Check for CUDA errors
     cudaError_t error = cudaGetLastError();
     if (error != cudaSuccess) {
         printf("CUDA Error: %s\n", cudaGetErrorString(error));
     }
     
     // Run additional test with increasing thread counts to test scalability
     printf("\nRunning scalability test with increasing thread counts...\n");
     
     int threadCounts[] = {64, 128, 256};
     for (int t = 0; t < 3; t++) {
         // Reset the queue
         initQueue(&h_queue);
         cudaMemcpy(d_queue, &h_queue, sizeof(MSQueue), cudaMemcpyHostToDevice);
         
         // Reset counters
         cudaMemcpy(d_enqueue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
         cudaMemcpy(d_dequeue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
         
         // Calculate blocks and threads
         int tpb = 64;
         int blocks = threadCounts[t] / tpb;
         if (blocks == 0) blocks = 1;
         
         printf("\nTesting with %d threads (%d blocks x %d threads)...\n", 
                blocks * tpb, blocks, tpb);
         
         // Launch kernel
         concurrentOperationsKernel<<<blocks, tpb>>>(
             d_queue, d_results, d_enqueue_count, d_dequeue_count, 3);
         cudaDeviceSynchronize();
         
         // Copy results
         cudaMemcpy(&h_enqueue_count, d_enqueue_count, sizeof(int), cudaMemcpyDeviceToHost);
         cudaMemcpy(&h_dequeue_count, d_dequeue_count, sizeof(int), cudaMemcpyDeviceToHost);
         
         // Print results
         printf("Results with %d threads: Enqueued %d, Dequeued %d\n", 
                blocks * tpb, h_enqueue_count, h_dequeue_count);
     }
     
     // Cleanup
     delete[] h_results;
     cudaFree(d_results);
     cudaFree(d_enqueue_count);
     cudaFree(d_dequeue_count);
     cudaFree(d_queue);
     
     return 0;
 }