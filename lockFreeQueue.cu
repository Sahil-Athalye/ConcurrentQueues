/**
 * Michael-Scott Lock-Free Queue Implementation in CUDA
 * Based on "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms"
 * 
 * This implementation provides a lock-free concurrent queue suitable for GPU environments.
 * The queue uses a linked list with head and tail pointers, with atomic operations to
 * ensure thread safety without locks.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 
 /**
  * Node structure with ABA problem prevention using a count field
  * Combined with pointer in the same 64-bit value
  */
 template<typename T>
 struct Node {
     T value;
     unsigned long long next;  // Lower bits for pointer, upper bits for counter
 };
 
 /**
  * Queue structure with head and tail pointers
  */
 template<typename T>
 struct MSQueue {
     unsigned long long head;  // Lower bits for pointer, upper bits for counter
     unsigned long long tail;  // Lower bits for pointer, upper bits for counter
     
     // The queue needs to be initialized before use
     __device__ __host__ void init() {
         // Create a dummy node
         Node<T>* dummy = new Node<T>();
         dummy->next = 0;  // nullptr with count = 0
         
         // Both head and tail point to the dummy node initially
         // Node pointer in lower bits, counter in upper bits
         unsigned long long node_ptr = reinterpret_cast<unsigned long long>(dummy);
         head = node_ptr;  // count = 0
         tail = node_ptr;  // count = 0
     }
     
     __device__ __host__ void cleanup() {
         // Free all nodes in the queue
         Node<T>* current = reinterpret_cast<Node<T>*>(head & 0xFFFFFFFFULL);
         while (current != nullptr) {
             Node<T>* next = reinterpret_cast<Node<T>*>(current->next & 0xFFFFFFFFULL);
             delete current;
             current = next;
         }
     }
 };
 
 /**
  * Helper functions for manipulating the combined pointer-counter values
  */
 __device__ __host__ inline Node<int>* getPointer(unsigned long long combined) {
     return reinterpret_cast<Node<int>*>(combined & 0xFFFFFFFFULL);
 }
 
 __device__ __host__ inline unsigned int getCounter(unsigned long long combined) {
     return static_cast<unsigned int>(combined >> 32);
 }
 
 __device__ __host__ inline unsigned long long makeTagged(Node<int>* ptr, unsigned int count) {
     return (static_cast<unsigned long long>(count) << 32) | reinterpret_cast<unsigned long long>(ptr);
 }
 
 /**
  * Enqueue an item to the queue
  * 
  * @param queue The queue to enqueue to
  * @param value The value to enqueue
  * @return True if enqueue was successful
  */
 template<typename T>
 __device__ bool enqueue(MSQueue<T>* queue, T value) {
     // Allocate a new node and initialize it
     Node<T>* node = new Node<T>();
     if (node == nullptr) return false;
     
     node->value = value;
     node->next = 0;  // null pointer with counter = 0
     
     unsigned long long tail, next;
     Node<T>* tail_ptr;
     Node<T>* next_ptr;
     
     while (true) {
         // Read the tail and its next pointer
         tail = atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail), 0ULL);
         tail_ptr = getPointer(tail);
         next = atomicAdd(reinterpret_cast<unsigned long long*>(&tail_ptr->next), 0ULL);
         next_ptr = getPointer(next);
         
         // Check if tail is consistent
         unsigned long long current_tail = atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail), 0ULL);
         if (tail != current_tail) continue;
         
         if (next_ptr == nullptr) {
             // Try to link the node at the end of the linked list
             unsigned long long new_next = makeTagged(node, getCounter(next) + 1);
             if (atomicCAS(reinterpret_cast<unsigned long long*>(&tail_ptr->next), 
                           next, new_next) == next) {
                 // Node is linked, try to advance tail
                 unsigned long long new_tail = makeTagged(node, getCounter(tail) + 1);
                 atomicCAS(reinterpret_cast<unsigned long long*>(&queue->tail), 
                           tail, new_tail);
                 return true;
             }
         } else {
             // Tail was not pointing to the last node, try to advance it
             unsigned long long new_tail = makeTagged(next_ptr, getCounter(tail) + 1);
             atomicCAS(reinterpret_cast<unsigned long long*>(&queue->tail), 
                       tail, new_tail);
         }
     }
 }
 
 /**
  * Dequeue an item from the queue
  * 
  * @param queue The queue to dequeue from
  * @param value Pointer to store the dequeued value
  * @return True if dequeue was successful, false if queue was empty
  */
 template<typename T>
 __device__ bool dequeue(MSQueue<T>* queue, T* value) {
     unsigned long long head, tail, next;
     Node<T>* head_ptr;
     Node<T>* tail_ptr;
     Node<T>* next_ptr;
     
     while (true) {
         // Read the head, tail, and head's next pointer
         head = atomicAdd(reinterpret_cast<unsigned long long*>(&queue->head), 0ULL);
         tail = atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail), 0ULL);
         head_ptr = getPointer(head);
         next = atomicAdd(reinterpret_cast<unsigned long long*>(&head_ptr->next), 0ULL);
         next_ptr = getPointer(next);
         
         // Check if head is consistent
         unsigned long long current_head = atomicAdd(reinterpret_cast<unsigned long long*>(&queue->head), 0ULL);
         if (head != current_head) continue;
         
         if (head_ptr == getPointer(tail)) {
             // Queue might be empty
             if (next_ptr == nullptr) {
                 // Queue is empty
                 return false;
             }
             
             // Tail is falling behind, try to advance it
             unsigned long long new_tail = makeTagged(next_ptr, getCounter(tail) + 1);
             atomicCAS(reinterpret_cast<unsigned long long*>(&queue->tail), 
                       tail, new_tail);
         } else {
             // No need to deal with tail
             // Read value before CAS
             if (next_ptr == nullptr) {
                 // This shouldn't happen
                 continue;
             }
             
             *value = next_ptr->value;
             
             // Try to advance head
             unsigned long long new_head = makeTagged(next_ptr, getCounter(head) + 1);
             if (atomicCAS(reinterpret_cast<unsigned long long*>(&queue->head), 
                           head, new_head) == head) {
                 // Successfully dequeued, free old dummy node
                 delete head_ptr;
                 return true;
             }
         }
     }
 }
 
 /**
  * CUDA kernel to test the queue
  */
 __global__ void testQueueKernel(MSQueue<int>* queue, int* results, int* enqueue_count, int* dequeue_count) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     // Even threads enqueue, odd threads dequeue
     if (tid % 2 == 0) {
         if (enqueue(queue, tid)) {
             atomicAdd(enqueue_count, 1);
         }
     } else {
         int value;
         if (dequeue(queue, &value)) {
             atomicAdd(dequeue_count, 1);
             // Store the dequeued value
             results[atomicAdd(dequeue_count, 0) - 1] = value;
         }
     }
 }
 
 /**
  * Main function to set up and run the test
  */
 int main() {
     // Allocate the queue on the device
     MSQueue<int>* d_queue;
     cudaMalloc(&d_queue, sizeof(MSQueue<int>));
     
     // Initialize the queue on the host
     MSQueue<int> h_queue;
     h_queue.init();
     
     // Copy the initialized queue to the device
     cudaMemcpy(d_queue, &h_queue, sizeof(MSQueue<int>), cudaMemcpyHostToDevice);
     
     // Allocate memory for results and counters
     int* d_results;
     int* d_enqueue_count;
     int* d_dequeue_count;
     cudaMalloc(&d_results, 1000 * sizeof(int));
     cudaMalloc(&d_enqueue_count, sizeof(int));
     cudaMalloc(&d_dequeue_count, sizeof(int));
     
     // Initialize counters
     int zero = 0;
     cudaMemcpy(d_enqueue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dequeue_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
     
     // Launch the kernel
     testQueueKernel<<<10, 100>>>(d_queue, d_results, d_enqueue_count, d_dequeue_count);
     
     // Wait for completion
     cudaDeviceSynchronize();
     
     // Copy back the results
     int h_enqueue_count, h_dequeue_count;
     int* h_results = new int[1000];
     cudaMemcpy(&h_enqueue_count, d_enqueue_count, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(&h_dequeue_count, d_dequeue_count, sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(h_results, d_results, h_dequeue_count * sizeof(int), cudaMemcpyDeviceToHost);
     
     // Print results
     printf("Enqueued: %d items\n", h_enqueue_count);
     printf("Dequeued: %d items\n", h_dequeue_count);
     
     // Cleanup
     delete[] h_results;
     cudaFree(d_results);
     cudaFree(d_enqueue_count);
     cudaFree(d_dequeue_count);
     
     // Cleanup the queue
     h_queue.cleanup();
     cudaFree(d_queue);
     
     return 0;
 }