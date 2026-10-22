// pcie_density_sweep.cu
//
// How much does sorting the rows of ONE step actually buy, at the density a
// step really has?
//
// The gather benchmark answers a related but different question. It sorts 4M
// indices over 11M rows — 36% density — where sorted order is nearly a
// sequential scan, and reports a 1.3-3.5x win. A training step is nowhere near
// that: a few tens of thousands of misses against hundreds of millions of rows,
// so 1e-4 or below. Sorted order there is monotonic but full of holes, and every
// access is still a fresh DRAM page and a fresh IOMMU translation. Whether the
// win survives is the question, and it decides whether sorting the miss batch
// (design doc P1.5) is worth doing at all.
//
// Two things this does that a naive sweep of the existing benchmark would get
// wrong:
//
//   * sorts WITHIN a step, not globally. That is what the cache would do: it
//     sees one step's misses, not the whole trace.
//   * walks a large index pool rather than reusing one small index set across
//     iterations. Reuse leaves the IOMMU TLB warm for exactly the pages being
//     measured, which flatters sparse access precisely where the answer matters.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -o pcie_density_sweep pcie_density_sweep.cu
// Run:   ./pcie_density_sweep [table_GB] [dim] [pool_M]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(_e));                               \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// One warp per row, as the cache's transfer path does.
__global__ void gather_rows(const float4* __restrict__ src,
                            float4* __restrict__ dst,
                            const uint32_t* __restrict__ idx, size_t n,
                            int vecs_per_row) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (size_t e = warp; e < n; e += nwarps) {
    const size_t s = static_cast<size_t>(idx[e]) * vecs_per_row;
    const size_t d = e * vecs_per_row;
    for (int off = lane; off < vecs_per_row; off += 32) dst[d + off] = src[s + off];
  }
}

int main(int argc, char** argv) {
  const double table_gb = (argc > 1 ? std::atof(argv[1]) : 16.0);
  const int dim = (argc > 2 ? std::atoi(argv[2]) : 384);   // 128 emb + Adam
  const double pool_m = (argc > 3 ? std::atof(argv[3]) : 4.0);

  if (dim % 4 != 0) { std::fprintf(stderr, "dim must be a multiple of 4\n"); return 1; }
  const int vecs = dim / 4;
  const size_t row_bytes = size_t(dim) * sizeof(float);
  const size_t table_bytes = size_t(table_gb * (1ull << 30));
  const size_t rows = table_bytes / row_bytes;
  const size_t pool = size_t(pool_m * 1e6);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
  std::printf("table  : %.1f GiB, %zu rows of %d floats (%zu B/row)\n",
              table_bytes / double(1ull << 30), rows, dim, row_bytes);
  std::printf("pool   : %zu row references per measurement (%.2f GiB moved)\n\n",
              pool, pool * row_bytes / double(1ull << 30));

  float4* h_src = nullptr;
  float4* d_src = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_src, table_bytes, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_src, h_src, 0));

  float4* d_dst = nullptr;  // device-resident sink: keeps one direction only

  std::mt19937 rng(20260807);
  std::uniform_int_distribution<uint32_t> dist(0, uint32_t(rows - 1));
  std::vector<uint32_t> idx(pool);
  for (size_t i = 0; i < pool; ++i) idx[i] = dist(rng);

  uint32_t* d_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, pool * sizeof(uint32_t)));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;

  std::printf("%12s %10s %14s %14s %8s\n", "rows/step", "density", "unsorted",
              "sorted", "gain");
  std::printf("%12s %10s %14s %14s %8s\n", "", "", "GiB/s", "GiB/s", "");

  // A step's miss count, from a few thousand up to where sorted order starts to
  // resemble a scan.
  const size_t steps_sizes[] = {4096,   16384,  65536,  262144,
                                1048576, 4194304};

  for (size_t per_step : steps_sizes) {
    if (per_step > pool) continue;
    const size_t nsteps = pool / per_step;
    const double density = double(per_step) / double(rows);
    double bw[2] = {0, 0};

    // The sink only needs to hold one step.
    CUDA_CHECK(cudaFree(d_dst));
    d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dst, per_step * row_bytes));

    for (int sorted = 0; sorted < 2; ++sorted) {
      std::vector<uint32_t> use(idx);
      if (sorted) {
        // Within each step only. The cache sees one step's misses; it has no
        // way to sort across steps.
        for (size_t s = 0; s < nsteps; ++s) {
          std::sort(use.begin() + s * per_step, use.begin() + (s + 1) * per_step);
        }
      }
      CUDA_CHECK(cudaMemcpy(d_idx, use.data(), pool * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));

      // Warm-up, then timed. Each step is its own launch over a different
      // slice, so no slice's translations stay resident for the next.
      for (size_t s = 0; s < std::min<size_t>(nsteps, 4); ++s) {
        gather_rows<<<grid, block>>>(d_src, d_dst, d_idx + s * per_step,
                                     per_step, vecs);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      cudaEvent_t t0, t1;
      CUDA_CHECK(cudaEventCreate(&t0));
      CUDA_CHECK(cudaEventCreate(&t1));
      CUDA_CHECK(cudaEventRecord(t0));
      for (size_t s = 0; s < nsteps; ++s) {
        gather_rows<<<grid, block>>>(d_src, d_dst, d_idx + s * per_step,
                                     per_step, vecs);
      }
      CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaEventSynchronize(t1));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
      CUDA_CHECK(cudaEventDestroy(t0));
      CUDA_CHECK(cudaEventDestroy(t1));

      const double gib = double(nsteps * per_step * row_bytes) / double(1ull << 30);
      bw[sorted] = gib / (ms / 1000.0);
    }

    std::printf("%12zu %9.4f%% %14.2f %14.2f %7.2fx\n", per_step,
                density * 100.0, bw[0], bw[1], bw[1] / bw[0]);
  }

  CUDA_CHECK(cudaFree(d_idx));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFreeHost(h_src));
  return 0;
}
