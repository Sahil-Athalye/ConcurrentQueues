/**
 * Improved Michael-Scott Lock-Free Queue Implementation in CUDA
 * 
 * Key improvements:
 * - Pre-allocated node pool instead of dynamic allocation
 * - Separated pointer and counter to avoid 64-bit atomic issues
 * - Added memory fences for better visibility between threads
 * - Simplified ABA prevention
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 
 // Constants for the queue
 #define QUEUE_SIZE 1024
 #define MAX_THREADS 1000
 
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
     while (true) {
         // Check if we have free nodes
         int freeList = atomicAdd((int*)&queue->freeList, 0);
         if (freeList == -1) {
             printf("No free nodes available\n");
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
 }
 
 /**
  * Return a node to the free list
  * 
  * @param queue The queue
  * @param nodeIndex Index of the node to free
  */
 __device__ void freeNode(MSQueue* queue, int nodeIndex) {
     while (true) {
         int currentFreeList = atomicAdd((int*)&queue->freeList, 0);
         queue->nodes[nodeIndex].next = currentFreeList;
         if (atomicCAS((int*)&queue->freeList, currentFreeList, nodeIndex) == currentFreeList) {
             atomicAdd((int*)&queue->freeCount, 1);
             return;
         }
     }
 }
 
 /**
  * Enqueue an item to the queue
  * 
  * @param queue The queue to enqueue to
  * @param value The value to enqueue
  * @return 0 on success, FULL if queue is full
  */
 __device__ int enqueue(MSQueue* queue, int value) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     printf("Thread %d: Attempting to enqueue value %d\n", tid, value);
     
     // Get a new node from the pool
     int nodeIndex = allocateNode(queue);
     if (nodeIndex == -1) {
         printf("Thread %d: Queue is full, no free nodes\n", tid);
         return FULL;
     }
     
     printf("Thread %d: Allocated node at index %d\n", tid, nodeIndex);
     
     // Initialize the new node
     queue->nodes[nodeIndex].value = value;
     queue->nodes[nodeIndex].next = -1;
     __threadfence();  // Ensure the node initialization is visible to other threads
     
     int attempts = 0;
     while (true) {
         attempts++;
         if (attempts > 1000) {
             printf("Thread %d: Giving up after %d attempts\n", tid, attempts);
             freeNode(queue, nodeIndex);
             return FULL;
         }
         
         // Read the current tail
         int tail = atomicAdd((int*)&queue->tail, 0);
         int tailCount = atomicAdd((int*)&queue->tailCount, 0);
         printf("Thread %d: Read tail=%d, tailCount=%d\n", tid, tail, tailCount);
         
         // Read the next pointer of the tail node
         int next = atomicAdd((int*)&queue->nodes[tail].next, 0);
         int nextCount = atomicAdd((int*)&queue->nodes[tail].nextCount, 0);
         printf("Thread %d: Read next=%d, nextCount=%d\n", tid, next, nextCount);
         
         // Check if tail is still consistent
         if (tail != atomicAdd((int*)&queue->tail, 0) || 
             tailCount != atomicAdd((int*)&queue->tailCount, 0)) {
             printf("Thread %d: Tail changed, retrying\n", tid);
             continue;
         }
         
         if (next == -1) {
             // Try to link the new node
             printf("Thread %d: Attempting CAS on tail next from %d to %d\n", tid, next, nodeIndex);
             if (atomicCAS((int*)&queue->nodes[tail].next, -1, nodeIndex) == -1) {
                 // Successfully linked, increment the nextCount
                 atomicAdd((int*)&queue->nodes[tail].nextCount, 1);
                 __threadfence();  // Ensure the link is visible before advancing tail
                 
                 // Now try to advance the tail
                 printf("Thread %d: Attempting to advance tail from %d to %d\n", tid, tail, nodeIndex);
                 atomicCAS((int*)&queue->tail, tail, nodeIndex);
                 atomicAdd((int*)&queue->tailCount, 1);
                 
                 printf("Thread %d: Enqueue success\n", tid);
                 return SUCCESS;
             }
         } else {
             // Tail is not pointing to the last node, try to advance it
             printf("Thread %d: Tail not pointing to last node, trying to advance\n", tid);
             atomicCAS((int*)&queue->tail, tail, next);
             atomicAdd((int*)&queue->tailCount, 1);
         }
     }
 }
 
 /**
  * Dequeue an item from the queue
  * 
  * @param queue The queue to dequeue from
  * @param value Pointer to store the dequeued value
  * @return 0 on success, EMPTY if queue was empty
  */
  __device__ int dequeue(MSQueue* queue, int* value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (true) {
        // Read head
        int head = atomicAdd((int*)&queue->head, 0);
        // Read next
        int next = atomicAdd((int*)&queue->nodes[head].next, 0);
        
        // If empty
        if (next == -1) {
            return EMPTY;
        }
        
        // Try to update head
        if (atomicCAS((int*)&queue->head, head, next) == head) {
            // Success - get value
            *value = queue->nodes[next].value;
            // Free the old node
            freeNode(queue, head);
            return SUCCESS;
        }
    }
}
 
 /**
  * CUDA kernel to test the queue
  */
 __global__ void testQueueKernel(MSQueue* queue, int* results, int* enqueue_count, int* dequeue_count) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     printf("Thread %d starting\n", tid);
     
     // Even threads enqueue, odd threads dequeue
     if (tid % 2 == 0) {
         printf("Thread %d attempting to enqueue %d\n", tid, tid);
         if (enqueue(queue, tid) == SUCCESS) {
             int count = atomicAdd(enqueue_count, 1);
             printf("Thread %d successfully enqueued %d (total enqueues: %d)\n", tid, tid, count + 1);
         } else {
             printf("Thread %d failed to enqueue %d\n", tid, tid);
         }
     } else {
         int value;
         printf("Thread %d attempting to dequeue\n", tid);
         if (dequeue(queue, &value) == SUCCESS) {
             int count = atomicAdd(dequeue_count, 1);
             printf("Thread %d successfully dequeued %d (total dequeues: %d)\n", tid, value, count + 1);
             // Store the dequeued value
             results[count] = value;
         } else {
             printf("Thread %d failed to dequeue\n", tid);
         }
     }
     
     printf("Thread %d finished\n", tid);
 }
 
/**
 * Enqueue-only kernel
 */
 __global__ void enqueueKernel(MSQueue* queue, int* enqueue_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    printf("Enqueue kernel: Thread %d starting\n", tid);
    
    int result = enqueue(queue, tid);
    if (result == SUCCESS) {
        int count = atomicAdd(enqueue_count, 1);
        printf("Enqueue kernel: Thread %d successfully enqueued %d (total enqueues: %d)\n", 
               tid, tid, count + 1);
    } else {
        printf("Enqueue kernel: Thread %d failed to enqueue %d, result: %d\n", 
               tid, tid, result);
    }
    
    printf("Enqueue kernel: Thread %d finished\n", tid);
}

/**
 * Dequeue-only kernel
 */
__global__ void dequeueKernel(MSQueue* queue, int* results, int* dequeue_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    printf("Dequeue kernel: Thread %d starting\n", tid);
    
    int value;
    int result = dequeue(queue, &value);
    if (result == SUCCESS) {
        int count = atomicAdd(dequeue_count, 1);
        printf("Dequeue kernel: Thread %d successfully dequeued %d (total dequeues: %d)\n", 
               tid, value, count + 1);
        // Store the dequeued value
        results[count] = value;
    } else {
        printf("Dequeue kernel: Thread %d failed to dequeue, result: %d\n", tid, result);
    }
    
    printf("Dequeue kernel: Thread %d finished\n", tid);
}

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
    
    // Set thread configuration
    int threadsPerBlock = 32;
    int numBlocks = 4;
    printf("Using %d blocks with %d threads each (%d total threads)\n", 
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);
    
    // Launch enqueue kernel first
    printf("Launching enqueue kernel...\n");
    enqueueKernel<<<numBlocks, threadsPerBlock>>>(d_queue, d_enqueue_count);
    cudaDeviceSynchronize();
    
    // Print intermediate results
    int h_enqueue_count_intermediate;
    cudaMemcpy(&h_enqueue_count_intermediate, d_enqueue_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nAfter enqueue phase:\n");
    printf("Enqueued: %d items\n", h_enqueue_count_intermediate);
    
    // Print queue state
    printf("\nPrinting queue state...\n");
    printQueueState<<<1, 1>>>(d_queue);
    cudaDeviceSynchronize();
    
    // Launch dequeue kernel after enqueues complete
    printf("\nLaunching dequeue kernel...\n");
    dequeueKernel<<<numBlocks, threadsPerBlock>>>(d_queue, d_results, d_dequeue_count);
    cudaDeviceSynchronize();
    
    // Copy back the final results
    int h_enqueue_count, h_dequeue_count;
    int* h_results = new int[QUEUE_SIZE];
    cudaMemcpy(&h_enqueue_count, d_enqueue_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dequeue_count, d_dequeue_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_dequeue_count > 0) {
        cudaMemcpy(h_results, d_results, h_dequeue_count * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Print final results
    printf("\nFinal Summary:\n");
    printf("Enqueued: %d items\n", h_enqueue_count);
    printf("Dequeued: %d items\n", h_dequeue_count);
    
    if (h_dequeue_count > 0) {
        printf("Dequeued values: ");
        for (int i = 0; i < min(h_dequeue_count, 20); i++) {
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
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_enqueue_count);
    cudaFree(d_dequeue_count);
    cudaFree(d_queue);
    
    return 0;
}