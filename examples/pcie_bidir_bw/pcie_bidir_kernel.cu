// pcie_bidir_kernel.cu
//
// Use a CUDA *kernel* (not the copy engines) to saturate PCIe in BOTH
// directions at once, via zero-copy access to mapped pinned host memory.
//
// Idea:
//   * Allocate host buffers with cudaHostAllocMapped so the GPU can address
//     them directly over PCIe (zero-copy / system memory).
//   * A kernel thread that READS from a host buffer generates H2D traffic;
//     a thread that WRITES to a host buffer generates D2H traffic.
//   * A kernel that reads host buffer A and writes host buffer B therefore
//     drives the downstream and upstream PCIe lanes *simultaneously* —
//     full-duplex, without ever touching cudaMemcpy.
//
// We compare three kernels over the same data volume:
//   read-only  : host -> (compute) -> device-resident sink   => H2D only
//   write-only : device-resident src -> (compute) -> host     => D2H only
//   bidir      : host -> (compute) -> host                    => H2D + D2H
//
// Latency to system memory is high, so we use lots of threads, a grid-stride
// loop, and float4 vectorized accesses to hide it.
//
// Build:  nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
//              -o pcie_bidir_kernel pcie_bidir_kernel.cu
// Run:    ./pcie_bidir_kernel [buffer_MB] [iterations]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Read host `src` over PCIe, add a constant, write to `dst`.
// Whether src/dst live in host or device memory decides the PCIe direction(s).
__global__ void stream_add(const float4* __restrict__ src,
                           float4* __restrict__ dst, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float4 v = src[i];                 // load  (H2D if src is host-mapped)
    v.x += 1.f; v.y += 1.f; v.z += 1.f; v.w += 1.f;
    dst[i] = v;                        // store (D2H if dst is host-mapped)
  }
}

int main(int argc, char** argv) {
  const size_t bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 256)
                       * 1024ull * 1024ull;               // default 256 MiB
  const int iters = (argc > 2 ? std::atoi(argv[2]) : 30);
  const size_t n = bytes / sizeof(float4);                // float4 elements

  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device %d: %s  (SMs=%d, canMapHostMemory=%d)\n",
         dev, prop.name, prop.multiProcessorCount, prop.canMapHostMemory);
  printf("  buffer = %.0f MiB, iterations = %d\n\n",
         bytes / (1024.0 * 1024.0), iters);

  // Host-mapped (zero-copy) buffers + their device-side pointers.
  float4 *h_a = nullptr, *h_b = nullptr, *d_ha = nullptr, *d_hb = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_a, bytes, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&h_b, bytes, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_ha, h_a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_hb, h_b, 0));

  // Device-resident buffers used as the "free" end that stays off the bus.
  float4 *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;   // heavily oversubscribe
  const double gib = bytes / (1024.0 * 1024.0 * 1024.0);

  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));

  auto run = [&](const char* label, const float4* src, float4* dst,
                 double buffers_on_bus) {
    stream_add<<<grid, block>>>(src, dst, n);          // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i)
      stream_add<<<grid, block>>>(src, dst, n);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    const double secs = ms / 1e3;
    // buffers_on_bus counts how many buffer-sized transfers crossed PCIe.
    const double bw = gib * buffers_on_bus * iters / secs;
    printf("  %-30s %7.2f GiB/s\n", label, bw);
    return bw;
  };

  printf("Kernel-driven PCIe bandwidth (zero-copy):\n");
  // read-only: read host h_a -> write device d_b  => 1 buffer H2D.
  const double rd = run("read-only  (H2D)",  d_ha, d_b, 1.0);
  // write-only: read device d_a -> write host h_b => 1 buffer D2H.
  const double wr = run("write-only (D2H)",  d_a,  d_hb, 1.0);
  // bidir: read host h_a -> write host h_b => 1 buffer H2D + 1 buffer D2H.
  const double bi = run("bidir      (H2D+D2H)", d_ha, d_hb, 2.0);

  printf("\nSum of one-way peaks : %.2f GiB/s\n", rd + wr);
  printf("Full-duplex measured : %.2f GiB/s\n", bi);
  printf("Duplex speedup vs best one-way : %.2fx\n",
         bi / (rd > wr ? rd : wr));

  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return 0;
}
