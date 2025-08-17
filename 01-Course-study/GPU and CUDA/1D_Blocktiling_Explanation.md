# Understanding 1D Blocktiling in CUDA SGEMM

This document explains the key concepts behind the 1D blocktiling kernel implementation for matrix multiplication (SGEMM) in CUDA.

## Key Template Parameters

From `runner.cu`, the kernel uses these parameters:
```cuda-cpp
const uint BM = 64;   // Block tile height (M dimension)
const uint BN = 64;   // Block tile width (N dimension) 
const uint BK = 8;    // Block tile depth (K dimension)
const uint TM = 8;    // Thread multiplier - elements per thread
```

## Thread Block Organization

### 1D Block Configuration
This kernel uses **1D thread blocks** with 512 threads per block:
```cuda-cpp
dim3 blockDim((BM * BN) / TM);  // = (64 * 64) / 8 = 512 threads
```

## Thread Index Mapping

### Linear to 2D Mapping
```cuda-cpp
const int threadCol = threadIdx.x % BN;  // Column position (0-63)
const int threadRow = threadIdx.x / BN;  // Row position (0-7)
```

**Key Insight**: `threadIdx.x` is just a linear thread identifier (0-511) and does NOT directly correspond to matrix columns. The kernel explicitly maps it to 2D coordinates.

### Example Mapping (BN = 8)
| threadIdx.x | threadCol (% BN) | threadRow (/ BN) | Matrix Position |
|-------------|------------------|------------------|-----------------|
| 0           | 0                | 0                | (row 0, col 0)  |
| 1           | 1                | 0                | (row 0, col 1)  |
| 7           | 7                | 0                | (row 0, col 7)  |
| 8           | 0                | 1                | (row 1, col 0)  |
| 15          | 7                | 1                | (row 1, col 7)  |

## Memory Loading Constraints

### Why `assert(BM * BK == blockDim.x)`?
This ensures there are **exactly enough threads** to load shared memory with one element per thread:

```cuda-cpp
// Each thread loads exactly one element into shared memory
As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
```

- **Shared memory size**: `As` has `BM * BK = 64 * 8 = 512` elements
- **Number of threads**: `blockDim.x = 512`
- **Loading pattern**: Each thread loads exactly 1 element

## Work Distribution per Thread

### Thread Multiplier (TM)
`TM = 8` means each thread computes **8 consecutive output elements** in the same column:

```cuda-cpp
float threadResults[TM] = {0.0};  // Each thread stores 8 results
```

### Thread Work Assignment
Thread with `threadRow = 1` and `TM = 8` handles matrix rows:
- Row `1*8 + 0 = 8`
- Row `1*8 + 1 = 9`
- Row `1*8 + 2 = 10`
- ...
- Row `1*8 + 7 = 15`

## Core Computation Loop

### Nested Loop Structure
```cuda-cpp
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  float tmpB = Bs[dotIdx * BN + threadCol];
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    threadResults[resIdx] +=
        As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
  }
}
```

### Loop Variables Explained
- **`dotIdx`** (0 to BK-1 = 0 to 7): Iterates through the K dimension for dot product computation
- **`resIdx`** (0 to TM-1 = 0 to 7): Iterates through the TM output elements this thread computes

### Performance Optimization: `tmpB`
`tmpB` caches a frequently reused value in a register:

**Without optimization** (inefficient):
```cuda-cpp
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
  threadResults[resIdx] += As[...] * Bs[dotIdx * BN + threadCol];  // Same memory access TM times!
}
```

**With optimization** (efficient):
```cuda-cpp
float tmpB = Bs[dotIdx * BN + threadCol];  // Load once into register
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
  threadResults[resIdx] += As[...] * tmpB;  // Reuse register value
}
```

## Matrix Element Addressing

### Why `threadRow * TM + resIdx`?
This calculates the actual matrix row for each output element:
- `threadRow * TM`: Starting row for this thread's block of elements
- `+ resIdx`: Offset within the TM rows (0, 1, 2, ..., 7)

### Example
Thread with `threadRow = 2`, `TM = 8`:
- `resIdx = 0`: row `2*8 + 0 = 16`
- `resIdx = 1`: row `2*8 + 1 = 17`
- `resIdx = 7`: row `2*8 + 7 = 23`

## Summary

The 1D blocktiling approach achieves high performance through:

1. **Efficient Memory Access**: Coalesced global memory access and shared memory reuse
2. **Work Distribution**: Each thread handles multiple (TM=8) output elements
3. **Register Optimization**: Caching frequently reused values like `tmpB`
4. **Flexible Mapping**: Linear thread indices mapped to 2D matrix coordinates
5. **Proper Synchronization**: Threads coordinate through shared memory and barriers

This design balances memory bandwidth utilization, computational throughput, and register usage to achieve efficient matrix multiplication on GPU hardware.
