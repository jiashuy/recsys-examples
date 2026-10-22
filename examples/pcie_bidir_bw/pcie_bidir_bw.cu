// pcie_bidir_bw.cu
//
// Demonstrate and measure PCIe *bidirectional* (full-duplex) bandwidth.
//
// Key facts this program exploits:
//   1. PCIe is full-duplex: the upstream (D2H) and downstream (H2D) lanes are
//      physically independent, so a read and a write can happen at the same time.
//   2. Modern NVIDIA GPUs expose separate copy engines for each direction
//      (see `asyncEngineCount`). To run both directions at once you must issue
//      the H2D copy and the D2H copy on *two different streams*.
//   3. Only *pinned* (page-locked) host memory can be DMA'd asynchronously.
//      Pageable memory forces a staging copy and serializes the transfer.
//
// So the recipe for using the full bidirectional bandwidth is:
//      pinned host buffers  +  two streams  +  concurrent cudaMemcpyAsync.
//
// Build:  nvcc -O3 -o pcie_bidir_bw pcie_bidir_bw.cu
// Run:    ./pcie_bidir_bw [buffer_MB] [iterations]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

using Clock = std::chrono::high_resolution_clock;

static double seconds_since(const Clock::time_point& t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
  const size_t bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 256)
                       * 1024ull * 1024ull;               // default 256 MiB
  const int iters = (argc > 2 ? std::atoi(argv[2]) : 20); // default 20 iters

  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device %d: %s\n", dev, prop.name);
  printf("  asyncEngineCount = %d  (>=2 means H2D and D2H can overlap)\n",
         prop.asyncEngineCount);
  printf("  buffer = %.0f MiB, iterations = %d\n\n",
         bytes / (1024.0 * 1024.0), iters);

  // --- Pinned host buffers: one that we upload, one that we download into. ---
  char *h_up = nullptr, *h_down = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_up,   bytes, cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h_down, bytes, cudaHostAllocDefault));

  // --- Device buffers: destination of the upload, source of the download. ---
  char *d_up = nullptr, *d_down = nullptr;
  CUDA_CHECK(cudaMalloc(&d_up,   bytes));
  CUDA_CHECK(cudaMalloc(&d_down, bytes));

  // Two streams so the two copy engines can run concurrently.
  cudaStream_t s_h2d, s_d2h;
  CUDA_CHECK(cudaStreamCreate(&s_h2d));
  CUDA_CHECK(cudaStreamCreate(&s_d2h));

  const double gib = bytes / (1024.0 * 1024.0 * 1024.0);

  auto run = [&](const char* label, bool do_h2d, bool do_d2h) {
    // Warmup.
    if (do_h2d) CUDA_CHECK(cudaMemcpyAsync(d_up, h_up, bytes,
                                           cudaMemcpyHostToDevice, s_h2d));
    if (do_d2h) CUDA_CHECK(cudaMemcpyAsync(h_down, d_down, bytes,
                                           cudaMemcpyDeviceToHost, s_d2h));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
      // Issue both directions before syncing so they overlap on the wire.
      if (do_h2d) CUDA_CHECK(cudaMemcpyAsync(d_up, h_up, bytes,
                                             cudaMemcpyHostToDevice, s_h2d));
      if (do_d2h) CUDA_CHECK(cudaMemcpyAsync(h_down, d_down, bytes,
                                             cudaMemcpyDeviceToHost, s_d2h));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double secs = seconds_since(t0);

    // Bytes actually moved across the link per iteration.
    const double gib_moved = gib * ((do_h2d ? 1 : 0) + (do_d2h ? 1 : 0));
    const double bw = gib_moved * iters / secs;
    printf("  %-28s %7.2f GiB/s\n", label, bw);
    return bw;
  };

  printf("Measured PCIe bandwidth:\n");
  const double h2d  = run("H2D only (upload)",       true,  false);
  const double d2h  = run("D2H only (download)",      false, true);
  const double both = run("H2D + D2H concurrent",     true,  true);

  printf("\nSum of one-way peaks : %.2f GiB/s\n", h2d + d2h);
  printf("Full-duplex measured : %.2f GiB/s\n", both);
  printf("Duplex speedup vs best one-way : %.2fx\n",
         both / (h2d > d2h ? h2d : d2h));

  CUDA_CHECK(cudaStreamDestroy(s_h2d));
  CUDA_CHECK(cudaStreamDestroy(s_d2h));
  CUDA_CHECK(cudaFree(d_up));
  CUDA_CHECK(cudaFree(d_down));
  CUDA_CHECK(cudaFreeHost(h_up));
  CUDA_CHECK(cudaFreeHost(h_down));
  return 0;
}
