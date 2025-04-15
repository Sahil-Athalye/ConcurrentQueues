# CUDA Concurrent Queue Implementations

This repository contains three different implementations of concurrent queues in CUDA, designed for high-performance parallel computing on GPUs:

1. **Lock-Free Queue (Michael-Scott Algorithm)**
2. **High-Throughput Blocking Queue**
3. **Flat-Combining Queue**

Each implementation demonstrates different approaches to concurrent data structures, with varying trade-offs between progress guarantees, throughput, and latency.

## Implementations

### Lock-Free Queue (Michael-Scott Algorithm)

Based on the seminal paper by Michael and Scott, this implementation provides a non-blocking, lock-free queue that uses atomic operations to ensure thread safety without locks. The algorithm uses a linked list structure with head and tail pointers and includes mechanisms to handle the ABA problem through versioning.

**Key characteristics:**
- Non-blocking and lock-free
- Guarantees that at least one thread makes progress
- Uses Compare-And-Swap (CAS) operations for synchronization

### High-Throughput Blocking Queue

Designed for many-core architectures, this array-based queue implementation is optimized for high throughput in environments with many concurrent threads. It uses fetch-and-add operations to select array slots and provides both blocking and non-blocking interfaces.

**Key characteristics:**
- Exceptional throughput on GPUs with many threads
- Uses atomic Fetch-And-Add operations for ticket acquisition
- Provides both blocking and non-blocking interfaces
- Includes status inspection capabilities

### Flat-Combining Queue

This implementation takes a fundamentally different approach where multiple operations are batched and processed by a single "combiner" thread. Threads publish their operations in publication records, reducing contention on the core data structure.

**Key characteristics:**
- Reduces contention by batching operations
- Single thread processes multiple operations at once
- Good performance under high contention
- Throughput limited by processing capacity of the combiner

## Building and Running

### Prerequisites

- NVIDIA CUDA Toolkit
- Compatible NVIDIA GPU
- Up-to-date GPU drivers

### Compilation

You can compile all three implementations using the provided build script:

```bash
# Make the build script executable
chmod +x buildPrograms.sh

# Run the build script
./buildPrograms.sh
```

Or compile them individually:

```bash
# Compile the lock-free queue
nvcc -o lockFree lockFreeQueue.cu

# Compile the high-throughput blocking queue
nvcc -o blocking blockingQueue.cu

# Compile the flat-combining queue
nvcc -o flatCombining flatCombiningQueue.cu
```

### Running

After compilation, execute each binary:

```bash
./lockFree
./blocking
./flatCombining
```

## Performance Considerations

- **Lock-Free Queue**: Provides strong progress guarantees but may suffer from contention with high thread counts.
- **Blocking Queue**: Offers highest throughput on GPUs with many threads but allows blocking.
- **Flat-Combining Queue**: Good performance under contention but maximum throughput limited by single-thread processing.

## References

1. Michael, M. M., & Scott, M. L. (1996). "Simple, fast, and practical non-blocking and blocking concurrent queue algorithms."
2. Scogland, T. R. W., & Feng, W. (2015). "Design and evaluation of scalable concurrent queues for many-core architectures."
3. Hendler, D., Incze, I., Shavit, N., & Tzafrir, M. (2010). "Flat combining and the synchronization-parallelism tradeoff."

## License

[Specify your license here]
