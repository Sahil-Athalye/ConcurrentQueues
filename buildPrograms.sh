mkdir -p debug
# Compile the lock-free queue (Michael-Scott)
nvcc -o ./debug/lockFree ./src/lockFreeQueue.cu

# Compile the high-throughput blocking queue
nvcc -o ./debug/blocking ./src/blockingQueue.cu

# Compile the flat-combining queue
nvcc -o ./debug/flatCombining ./src/flatCombiningQueue.cu