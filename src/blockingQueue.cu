/**
 * High-Throughput Blocking Array-Based Queue in CUDA
 * Based on "Design and Evaluation of Scalable Concurrent Queues for Many-Core Architectures"
 * 
 * This implementation provides a blocking concurrent queue optimized for high throughput
 * in many-core environments. It uses fetch-and-add operations to manage queue indices.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 
 // Define maximum values to handle integer rollover
 #define MAX_ID (UINT_MAX/(QUEUE_SIZE*2))
 #define MAX_THREADS 1024
 #define QUEUE_SIZE 1024
 #define MAX_DISTANCE (QUEUE_SIZE + MAX_THREADS)
 
 // Return codes
 #define SUCCESS 0
 #define CLOSED 1
 #define BUSY 2
 
 // Get the ID for a given ticket
 __device__ __host__ inline unsigned int GET_ID(unsigned int x) {
     return ((x / QUEUE_SIZE) * 2);
 }
 
 // Safely increment an ID (handles rollover)
 __device__ inline void INC_SAFE(volatile unsigned int* ids, unsigned int target, unsigned int id) {
     atomicExch((unsigned int*)&ids[target], ((id+1) % MAX_ID));
 }
 
 /**
  * Queue structure with head and tail indices
  */
 struct HTQueue {
     volatile unsigned int head;
     volatile unsigned int tail;
     volatile int closed;
     volatile unsigned int items[QUEUE_SIZE];
     volatile unsigned int ids[QUEUE_SIZE];
 };
 
 /**
  * Initialize a queue
  */
 __host__ void initQueue(HTQueue* queue) {
     queue->head = 0;
     queue->tail = 0;
     queue->closed = 0;
     
     // Initialize all IDs to 0
     for (int i = 0; i < QUEUE_SIZE; i++) {
         queue->ids[i] = 0;
     }
 }
 
 /**
  * Enqueue an item (blocking)
  */
 __device__ int enqueue(HTQueue* queue, unsigned int item) {
     // Check if queue is closed
     if (atomicAdd((int*)&queue->closed, 0) != 0) {
         return CLOSED;
     }
     
     // Get a ticket by atomically incrementing tail
     unsigned int ticket = atomicAdd((unsigned int*)&queue->tail, 1);
     unsigned int target = ticket % QUEUE_SIZE;
     unsigned int id = GET_ID(ticket);
     
     // Wait until our slot is available (id matches expected id)
     while (atomicAdd((unsigned int*)&queue->ids[target], 0) != id) {
         // Check if queue was closed while waiting
         if (atomicAdd((int*)&queue->closed, 0) != 0) {
             // Rollback the ticket
             atomicSub((unsigned int*)&queue->tail, 1);
             return CLOSED;
         }
         
         // Simple exponential backoff
         for (int i = 0; i < 32; i++) {
             __threadfence();
         }
     }
     
     // Place the item in the queue
     atomicExch((unsigned int*)&queue->items[target], item);
     
     // Update the ID to allow the next operation on this slot
     INC_SAFE(queue->ids, target, id);
     
     return SUCCESS;
 }
 
 /**
  * Dequeue an item (blocking)
  */
 __device__ int dequeue(HTQueue* queue, unsigned int* item) {
     // Check if queue is closed
     if (atomicAdd((int*)&queue->closed, 0) != 0) {
         return CLOSED;
     }
     
     // Get a ticket by atomically incrementing head
     unsigned int ticket = atomicAdd((unsigned int*)&queue->head, 1);
     unsigned int target = ticket % QUEUE_SIZE;
     unsigned int id = GET_ID(ticket) + 1; // +1 for dequeue ID
     
     // Wait until our slot is available (id matches expected id)
     while (atomicAdd((unsigned int*)&queue->ids[target], 0) != id) {
         // Check if queue was closed while waiting
         if (atomicAdd((int*)&queue->closed, 0) != 0) {
             // Rollback the ticket
             atomicSub((unsigned int*)&queue->head, 1);
             return CLOSED;
         }
         
         // Simple exponential backoff
         for (int i = 0; i < 32; i++) {
             __threadfence();
         }
     }
     
     // Get the item from the queue
     *item = atomicAdd((unsigned int*)&queue->items[target], 0);
     
     // Update the ID to allow the next operation on this slot
     INC_SAFE(queue->ids, target, id);
     
     return SUCCESS;
 }
 
 /**
  * Enqueue an item (non-waiting)
  */
 __device__ int enqueue_nb(HTQueue* queue, unsigned int item) {
     // Check if queue is closed
     if (atomicAdd((int*)&queue->closed, 0) != 0) {
         return CLOSED;
     }
     
     // Read the current tail value
     unsigned int ticket = atomicAdd((unsigned int*)&queue->tail, 0);
     unsigned int target = ticket % QUEUE_SIZE;
     unsigned int id = GET_ID(ticket);
     
     // Check if the target slot is ready
     if (atomicAdd((unsigned int*)&queue->ids[target], 0) != id) {
         return BUSY; // Next slot not ready
     }
     
     // Try to get the ticket using CAS
     if (atomicCAS((unsigned int*)&queue->tail, ticket, ticket+1) != ticket) {
         return BUSY; // CAS failed, return
     }
     
     // Place the item in the queue
     atomicExch((unsigned int*)&queue->items[target], item);
     
     // Update the ID to allow the next operation on this slot
     INC_SAFE(queue->ids, target, id);
     
     return SUCCESS;
 }
 
 /**
  * Dequeue an item (non-waiting)
  */
 __device__ int dequeue_nb(HTQueue* queue, unsigned int* item) {
     // Check if queue is closed
     if (atomicAdd((int*)&queue->closed, 0) != 0) {
         return CLOSED;
     }
     
     // Read the current head value
     unsigned int ticket = atomicAdd((unsigned int*)&queue->head, 0);
     unsigned int target = ticket % QUEUE_SIZE;
     unsigned int id = GET_ID(ticket) + 1; // +1 for dequeue ID
     
     // Check if the target slot is ready
     if (atomicAdd((unsigned int*)&queue->ids[target], 0) != id) {
         return BUSY; // Next slot not ready
     }
     
     // Try to get the ticket using CAS
     if (atomicCAS((unsigned int*)&queue->head, ticket, ticket+1) != ticket) {
         return BUSY; // CAS failed, return
     }
     
     // Get the item from the queue
     *item = atomicAdd((unsigned int*)&queue->items[target], 0);
     
     // Update the ID to allow the next operation on this slot
     INC_SAFE(queue->ids, target, id);
     
     return SUCCESS;
 }
 
 /**
  * Close the queue - all waiting operations will return CLOSED
  */
 __device__ void closeQueue(HTQueue* queue) {
     atomicExch((int*)&queue->closed, 1);
 }
 
 /**
  * Check the distance between head and tail (number of items in queue)
  */
 __device__ int getDistance(HTQueue* queue) {
     unsigned int head = atomicAdd((unsigned int*)&queue->head, 0);
     unsigned int tail = atomicAdd((unsigned int*)&queue->tail, 0);
     
     if (tail >= head) {
         return tail - head;
     } else {
         // Handle counter rollover
         if (tail + MAX_DISTANCE < head) {
             // tail has rolled over
             return (UINT_MAX - head) + tail + 1;
         } else {
             // head has rolled over
             return tail - head;
         }
     }
 }
 
 /**
  * Check if the queue is empty
  */
 __device__ bool isEmpty(HTQueue* queue) {
     return getDistance(queue) == 0;
 }
 
 /**
  * Check if the queue is full
  */
 __device__ bool isFull(HTQueue* queue) {
     return getDistance(queue) >= QUEUE_SIZE;
 }
 
 /**
  * CUDA kernel to test both blocking and non-blocking methods
  */
 __global__ void testQueueKernel(HTQueue* queue, int* enqueueSuccess, int* dequeueSuccess, 
                                  unsigned int* results, bool useBlocking) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     // Determine operation based on thread ID
     if (tid % 2 == 0) {
         // Enqueue operation - use the thread ID as the value
         int status;
         if (useBlocking) {
             status = enqueue(queue, tid);
         } else {
             status = enqueue_nb(queue, tid);
         }
         
         if (status == SUCCESS) {
             atomicAdd(enqueueSuccess, 1);
         }
     } else {
         // Dequeue operation
         unsigned int value;
         int status;
         if (useBlocking) {
             status = dequeue(queue, &value);
         } else {
             status = dequeue_nb(queue, &value);
         }
         
         if (status == SUCCESS) {
             int idx = atomicAdd(dequeueSuccess, 1);
             if (idx < QUEUE_SIZE) {
                 results[idx] = value;
             }
         }
     }
 }
 
 /**
  * Producer kernel (only enqueue)
  */
 __global__ void producerKernel(HTQueue* queue, int numItems, int* enqueueSuccess) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     if (tid < numItems) {
         int status = enqueue(queue, tid);
         if (status == SUCCESS) {
             atomicAdd(enqueueSuccess, 1);
         }
     }
 }
 
 /**
  * Consumer kernel (only dequeue)
  */
 __global__ void consumerKernel(HTQueue* queue, int numItems, int* dequeueSuccess, unsigned int* results) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     if (tid < numItems) {
         unsigned int value;
         int status = dequeue(queue, &value);
         if (status == SUCCESS) {
             int idx = atomicAdd(dequeueSuccess, 1);
             if (idx < numItems) {
                 results[idx] = value;
             }
         }
     }
 }
 
 /**
  * Main function to test the HTQueue
  */
 int main() {
     // Allocate queue on host and device
     HTQueue* d_queue;
     cudaMalloc(&d_queue, sizeof(HTQueue));
     
     // Initialize queue on the host
     HTQueue h_queue;
     initQueue(&h_queue);
     
     // Copy to device
     cudaMemcpy(d_queue, &h_queue, sizeof(HTQueue), cudaMemcpyHostToDevice);
     
     // Allocate memory for results and counters
     int* d_enqueueSuccess;
     int* d_dequeueSuccess;
     unsigned int* d_results;
     cudaMalloc(&d_enqueueSuccess, sizeof(int));
     cudaMalloc(&d_dequeueSuccess, sizeof(int));
     cudaMalloc(&d_results, QUEUE_SIZE * sizeof(unsigned int));
     
     // Initialize counters
     int zero = 0;
     cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Test blocking version
     printf("Testing blocking queue operations...\n");
     testQueueKernel<<<10, 100>>>(d_queue, d_enqueueSuccess, d_dequeueSuccess, d_results, true);
     cudaDeviceSynchronize();
     
     // Copy results back
     int h_enqueueSuccess, h_dequeueSuccess;
     unsigned int* h_results = new unsigned int[QUEUE_SIZE];
     cudaMemcpy(&h_enqueueSuccess, d_enqueueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeueSuccess, d_dequeueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(h_results, d_results, h_dequeueSuccess * sizeof(unsigned int), cudaMemcpyDeviceToHost);
     
     printf("Blocking test results:\n");
     printf("  Enqueue successes: %d\n", h_enqueueSuccess);
     printf("  Dequeue successes: %d\n", h_dequeueSuccess);
     
     // Reset counters for non-blocking test
     cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Test non-blocking version
     printf("\nTesting non-blocking queue operations...\n");
     testQueueKernel<<<10, 100>>>(d_queue, d_enqueueSuccess, d_dequeueSuccess, d_results, false);
     cudaDeviceSynchronize();
     
     // Copy results back
     cudaMemcpy(&h_enqueueSuccess, d_enqueueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeueSuccess, d_dequeueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(h_results, d_results, h_dequeueSuccess * sizeof(unsigned int), cudaMemcpyDeviceToHost);
     
     printf("Non-blocking test results:\n");
     printf("  Enqueue successes: %d\n", h_enqueueSuccess);
     printf("  Dequeue successes: %d\n", h_dequeueSuccess);
     
     // Producer-Consumer test
     printf("\nTesting Producer-Consumer pattern...\n");
     
     // Reset counters and queue
     cudaMemcpy(d_queue, &h_queue, sizeof(HTQueue), cudaMemcpyHostToDevice);
     cudaMemcpy(d_enqueueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeueSuccess, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Launch producer
     int numItems = 500;
     producerKernel<<<5, 100>>>(d_queue, numItems, d_enqueueSuccess);
     
     // Launch consumer
     consumerKernel<<<5, 100>>>(d_queue, numItems, d_dequeueSuccess, d_results);
     cudaDeviceSynchronize();
     
     // Copy results back
     cudaMemcpy(&h_enqueueSuccess, d_enqueueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeueSuccess, d_dequeueSuccess, sizeof(int), cudaMemcpyDeviceToHost);
     
     printf("Producer-Consumer test results:\n");
     printf("  Enqueue successes: %d\n", h_enqueueSuccess);
     printf("  Dequeue successes: %d\n", h_dequeueSuccess);
     
     // Cleanup
     delete[] h_results;
     cudaFree(d_results);
     cudaFree(d_enqueueSuccess);
     cudaFree(d_dequeueSuccess);
     cudaFree(d_queue);
     
     return 0;
 }