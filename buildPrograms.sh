mkdir -p debug
# Compile the lock-free queue (Michael-Scott)
# nvcc -std=c++17 -arch=sm_89 \
#      -gencode arch=compute_89,code=sm_89 \
#      -gencode arch=compute_89,code=compute_89 \
#         ./src/lockFreeQueue.cu -o ./debug/lockFree
nvcc -o ./debug/lockFree ./src/lockFreeQueue.cu

# Compile the high-throughput blocking queue
nvcc -std=c++17 -arch=sm_89 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89 \
        ./src/blockingQueue.cu -o ./debug/blocking

# Compile the flat-combining queue
nvcc -std=c++17 -arch=sm_89 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89 \
        ./src/flatCombiningQueue.cu -o ./debug/flatCombining
