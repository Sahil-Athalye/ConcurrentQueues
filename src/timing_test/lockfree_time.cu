/**
 * True Concurrent Michael-Scott Lock-Free Queue Implementation in CUDA
 * 
 * This version allows for simultaneous enqueue and dequeue operations
 * with proper handling of contention and memory ordering.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 #include <curand_kernel.h> // To generate random numbers in CUDA
 
// Constants for the queue
 #define QUEUE_SIZE 1024
 #define MAX_THREADS 1000
 #define MAX_RETRIES 10

// Constants for the queue
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
 * Concurrent kernel with a pre-defined list of enqueue and dequeue operations
 */
__global__ void concurrentOperationsKernel(MSQueue* queue, int* results, int* enqueue_count, int* dequeue_count, int* operation_sequence, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    printf("op");
    for (int i = 0; i < iterations; i++) {
        // Get the operation from the list
        int op = operation_sequence[tid * iterations + i]; // Each thread uses the list to perform the correct operation

        if (op == 0) {
            // Enqueue operation
            int value = tid * 1000 + i; // Create a unique value
            if (enqueue(queue, value) == SUCCESS) {
                atomicAdd(enqueue_count, 1);
            }
        } else {
            // Dequeue operation
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
 * Main function to set up and run the test with pre-defined operations
 */

/**
 * Main function to set up and run the test with pre-defined operations
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
    int* d_operation_sequence; // For storing the sequence of operations (enqueue or dequeue)
    cudaMalloc(&d_results, QUEUE_SIZE * sizeof(int));
    cudaMalloc(&d_enqueue_count, sizeof(int));
    cudaMalloc(&d_dequeue_count, sizeof(int));
    cudaMalloc(&d_operation_sequence, sizeof(int) * 1000 * 32); // 1000 operations for 32 threads

    // Initialize counters
    int zero = 0;
    cudaMemcpy(d_enqueue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dequeue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Generate the list of operations on the host (0 for enqueue, 1 for dequeue)
    int* h_operation_sequence = new int[1000 * 32]; // 1000 operations for 32 threads
    for (int i = 0; i < 1000 * 32; i++) {
        h_operation_sequence[i] = rand() % 2; // Randomly assign 0 (enqueue) or 1 (dequeue)
    }

    // Copy the list of operations to the device
    cudaMemcpy(d_operation_sequence, h_operation_sequence, sizeof(int) * 1000 * 32, cudaMemcpyHostToDevice);

    // Set up the test parameters
    int iterations = 1000; // Each thread performs this many operations
    int threadsPerBlock = 32;

    // Loop through different block configurations
    for (int numBlocks = 1; numBlocks <= 8; numBlocks++) {
        // Test with the current configuration
        int totalThreads = threadsPerBlock * numBlocks;

        // Record start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Launch the kernel
        concurrentOperationsKernel<<<numBlocks, threadsPerBlock>>>(d_queue, d_results, d_enqueue_count, d_dequeue_count, d_operation_sequence, iterations);

	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
        // Synchronize device to ensure kernel execution completes before stopping the timer
        cudaDeviceSynchronize();

        // Record stop time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Make sure the event is fully recorded

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Output the results
        printf("Configuration: %d blocks, %d threads per block\n", numBlocks, threadsPerBlock);
        printf("Time taken: %f ms\n", milliseconds);

        // Clean up
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    delete[] h_operation_sequence;
    cudaFree(d_results);
    cudaFree(d_enqueue_count);
    cudaFree(d_dequeue_count);
    cudaFree(d_operation_sequence);
    cudaFree(d_queue);

    return 0;
}

