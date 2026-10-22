// pcie_bidir_gather.cu
//
// Indirect (gather/scatter) variant of the kernel-driven PCIe bidirectional
// bandwidth test — modeled on embedding-table access.
//
//   * Two large host tables (tens of GB each) live in mapped pinned memory,
//     because they are far too big for device memory: the kernel reaches them
//     over PCIe via zero-copy.
//   * Random src_idx[] / dst_idx[] scatter the accessed rows across the whole
//     table, so this is genuine indirect access, not a contiguous stream.
//   * Each warp copies one row: gather row src_idx[e] from the source table,
//     scatter it to row dst_idx[e] of the destination table. Reading the
//     source drives H2D; writing the destination drives D2H — both at once.
//
// Three configurations over the same row volume:
//   gather-only  : host src_table[rand] -> contiguous device buffer  => H2D
//   scatter-only : contiguous device buffer -> host dst_table[rand]   => D2H
//   bidir        : host src_table[rand] -> host dst_table[rand]       => H2D+D2H
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
//             -o pcie_bidir_gather pcie_bidir_gather.cu
// Run:   ./pcie_bidir_gather [table_GB_each] [entries_M] [dim] [iters]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// One warp per entry. A null index array means "contiguous" (row == entry),
// which keeps that side off the PCIe bus (device-resident buffer).
__global__ void gather_scatter(const float4* __restrict__ src_table,
                               float4* __restrict__ dst_table,
                               const uint32_t* __restrict__ src_idx,
                               const uint32_t* __restrict__ dst_idx,
                               size_t num_entries, int vecs_per_row) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane    = threadIdx.x & 31;
  const int nwarps  = (gridDim.x * blockDim.x) >> 5;

  for (size_t e = warp_id; e < num_entries; e += nwarps) {
    const size_t s = src_idx ? (size_t)src_idx[e] : e;   // random gather / contiguous
    const size_t d = dst_idx ? (size_t)dst_idx[e] : e;   // random scatter / contiguous
    const size_t sbase = s * vecs_per_row;
    const size_t dbase = d * vecs_per_row;
    for (int off = lane; off < vecs_per_row; off += 32)
      dst_table[dbase + off] = src_table[sbase + off];   // read + write across PCIe
  }
}

int main(int argc, char** argv) {
  const double table_gb = (argc > 1 ? std::atof(argv[1]) : 16.0); // per table
  const size_t entries  = (size_t)((argc > 2 ? std::atof(argv[2]) : 4.0) * 1e6);
  const int dim         = (argc > 3 ? std::atoi(argv[3]) : 128);  // floats/row
  const int iters       = (argc > 4 ? std::atoi(argv[4]) : 10);
  const int do_sort     = (argc > 5 ? std::atoi(argv[5]) : 0);    // sort indices?

  if (dim % 4 != 0) { fprintf(stderr, "dim must be a multiple of 4\n"); return 1; }
  const int vecs_per_row = dim / 4;                       // float4 per row
  const size_t row_bytes = (size_t)dim * sizeof(float);
  const size_t table_bytes = (size_t)(table_gb * (1ull << 30));
  const size_t rows = table_bytes / row_bytes;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
  printf("  table   = %.1f GiB each, %zu rows of %d floats (%zu B/row)\n",
         table_bytes / (double)(1ull << 30), rows, dim, row_bytes);
  printf("  entries = %zu (%.1f M),  moved/dir = %.2f GiB,  iters = %d\n\n",
         entries, entries / 1e6,
         entries * row_bytes / (double)(1ull << 30), iters);

  // --- Large mapped pinned host tables (zero-copy) ---
  printf("Allocating %.1f GiB pinned host memory (x2)...\n", table_bytes / (double)(1ull<<30));
  float4 *h_src = nullptr, *h_dst = nullptr, *d_hsrc = nullptr, *d_hdst = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_src, table_bytes, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&h_dst, table_bytes, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_hsrc, h_src, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_hdst, h_dst, 0));

  // Contiguous device-resident staging buffer for the single-direction cases.
  float4 *d_stage = nullptr;
  CUDA_CHECK(cudaMalloc(&d_stage, entries * row_bytes));

  // --- Random indices on host, copied to device ---
  printf("Generating %zu random indices...\n", entries);
  std::vector<uint32_t> h_si(entries), h_di(entries);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)(rows - 1));
  for (size_t i = 0; i < entries; ++i) { h_si[i] = dist(rng); h_di[i] = dist(rng); }
  if (do_sort) {                     // sort -> adjacent entries hit adjacent rows
    std::sort(h_si.begin(), h_si.end());
    std::sort(h_di.begin(), h_di.end());
    printf("Indices SORTED (ascending) for locality.\n");
  } else {
    printf("Indices RANDOM (unsorted).\n");
  }
  uint32_t *d_si = nullptr, *d_di = nullptr;
  CUDA_CHECK(cudaMalloc(&d_si, entries * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_di, entries * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_si, h_si.data(), entries * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_di, h_di.data(), entries * sizeof(uint32_t), cudaMemcpyHostToDevice));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;
  const double gib_per_dir = entries * row_bytes / (double)(1ull << 30);

  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));

  auto run = [&](const char* label, const float4* st, float4* dt,
                 const uint32_t* si, const uint32_t* di, double dirs) {
    gather_scatter<<<grid, block>>>(st, dt, si, di, entries, vecs_per_row); // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i)
      gather_scatter<<<grid, block>>>(st, dt, si, di, entries, vecs_per_row);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    const double bw = gib_per_dir * dirs * iters / (ms / 1e3);
    printf("  %-26s %7.2f GiB/s\n", label, bw);
    return bw;
  };

  printf("\nIndirect (gather/scatter) PCIe bandwidth:\n");
  // gather-only : random-read host src -> contiguous device stage  (H2D)
  const double g = run("gather-only  (H2D)",  d_hsrc, d_stage, d_si, nullptr, 1.0);
  // scatter-only: contiguous device stage -> random-write host dst (D2H)
  const double s = run("scatter-only (D2H)",  d_stage, d_hdst, nullptr, d_di, 1.0);
  // bidir       : random host -> random host                       (H2D+D2H)
  const double b = run("bidir        (H2D+D2H)", d_hsrc, d_hdst, d_si, d_di, 2.0);

  printf("\nSum of one-way peaks : %.2f GiB/s\n", g + s);
  printf("Full-duplex measured : %.2f GiB/s\n", b);
  printf("Duplex speedup vs best one-way : %.2fx\n", b / (g > s ? g : s));

  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_stage));
  CUDA_CHECK(cudaFree(d_si));
  CUDA_CHECK(cudaFree(d_di));
  CUDA_CHECK(cudaFreeHost(h_src));
  CUDA_CHECK(cudaFreeHost(h_dst));
  return 0;
}
