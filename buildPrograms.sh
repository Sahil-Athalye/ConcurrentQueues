# Compile the lock-free queue (Michael-Scott)
nvcc -o lockFree lockFreeQueue.cu

# Compile the high-throughput blocking queue
nvcc -o blocking blockingQueue.cu

# Compile the flat-combining queue
nvcc -o flatCombining flatCombiningQueue.cu