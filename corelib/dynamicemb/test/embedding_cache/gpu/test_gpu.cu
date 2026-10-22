// ---------------------------------------------------------------------------
// embedding_cache on a device.
//
// Everything the host tests cannot reach: whether these headers compile with
// nvcc at all, and whether the protocol survives real contention. The host
// build runs against a stand-in for the CUDA runtime that serialises every
// atomic, so it can show a protocol is right and cannot show it is race-free.
//
// No embedding rows here either. Each node gets one int64 standing in for its
// row; allocate writes the key into it, and every hit checks it reads its own
// key back. That is enough to catch the two failures that matter — a node bound
// to two keys, and a key bound to two nodes — and it needs neither a row pool
// nor a backing store, neither of which exists yet.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#include "kernels.cuh"

using namespace embedding_cache;

static int failures = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("  FAIL: ");                    \
      std::printf(__VA_ARGS__);                   \
      std::printf("  (%s:%d)\n", __FILE__, __LINE__); \
      ++failures;                                 \
    }                                             \
  } while (0)

#define CU(call)                                                          \
  do {                                                                    \
    const cudaError_t e_ = (call);                                        \
    if (e_ != cudaSuccess) {                                              \
      std::printf("  CUDA FAIL: %s -> %s  (%s:%d)\n", #call,              \
                  cudaGetErrorString(e_), __FILE__, __LINE__);            \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// What the cache would define: the map's three fields from the base, the
// concurrency word from the other component.
struct Node : HashMapNode<int64_t, int32_t> {
  AtomicNodeState state;
};

using MapStore = HashMapStorage<Node>;
using Map = MapStore::Map;
using QStore = ReclaimQueueStorage<Map::NodeId, Map::kNoNode>;
using Queue = QStore::Queue;
using Slot = AllocSlot<Map::Key, Map::NodeId>;

static_assert(sizeof(Node) == 24, "key 8 + links 8 + state 4, padded to 24");
static_assert(Queue::kNoNode == Map::kNoNode, "one source of truth");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    CU(cudaMalloc(&p, sizeof(T) * count));
    CU(cudaMemset(p, 0, sizeof(T) * count));
  }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  std::vector<T> get() const {
    std::vector<T> out(n);
    CU(cudaMemcpy(out.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return out;
  }
  void put(const std::vector<T>& v) {
    CU(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  }
};

// Every node starts EMPTY and already in the queue: the queue is arange(C), so
// each node is sitting in it. HashMapStorage cannot do this — it does not know
// what a state word should hold — so it is the caller's, which here is us.
__global__ void init_states_kernel(Node* nodes, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    nodes[i].state.set(NodeState::make(NodeState::kEmpty, 0u,
                                       /*in_ring=*/true, /*used=*/false));
  }
}

// ---------------------------------------------------------------------------
// 1. NodeState under contention
// ---------------------------------------------------------------------------
//
// Many threads pin and release ONE node. Every pin that succeeds is matched by
// a release, so the count has to come back to where it started. A lost update
// in either direction shows up here and nowhere else.

__global__ void hammer_pin_release_kernel(Node* nodes, int32_t node, int rounds,
                                          unsigned long long* pins,
                                          unsigned long long* enqueues) {
  for (int r = 0; r < rounds; ++r) {
    if (nodes[node].state.try_pin()) {
      atomicAdd(pins, 1ull);
      if (nodes[node].state.release()) atomicAdd(enqueues, 1ull);
    }
  }
}

static void test_state_contention() {
  std::printf("=== NodeState under contention ===\n");
  const int64_t kNodes = 1;
  Dev<Node> nodes(kNodes);
  Dev<unsigned long long> pins(1), enqueues(1);

  std::vector<Node> h(1);
  h[0].key = 42;
  h[0].prev = h[0].next = Map::kNoNode;
  // Resident, one reference held by "the test", already in the queue.
  h[0].state.set(NodeState::make(NodeState::kResident, 1u, true, false));
  nodes.put(h);

  const int kBlocks = 64, kThreads = 256, kRounds = 64;
  hammer_pin_release_kernel<<<kBlocks, kThreads>>>(nodes.p, 0, kRounds, pins.p,
                                                   enqueues.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto after = nodes.get()[0].state.get();
  const auto npins = pins.get()[0];
  std::printf("  %llu pins by %d threads x %d rounds, refs back to %u\n",
              npins, kBlocks * kThreads, kRounds, after.refs());
  CHECK(npins == (unsigned long long)kBlocks * kThreads * kRounds,
        "every pin on a resident node should succeed");
  CHECK(after.refs() == 1u, "refs returned to its starting value, got %u",
        after.refs());
  CHECK(after.residency() == NodeState::kResident, "residency untouched");
  CHECK(after.used(), "pin recorded recency");
  CHECK(enqueues.get()[0] == 0ull,
        "the node never reached zero references, so nobody owned an enqueue");
}

// ---------------------------------------------------------------------------
// 2. try_claim hands one node to exactly one thread
// ---------------------------------------------------------------------------

__global__ void hammer_claim_kernel(Node* nodes, int64_t n,
                                    unsigned long long* wins) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n * 8; i += int64_t(gridDim.x) * blockDim.x) {
    NodeState was = NodeState::make(NodeState::kEmpty, 0u, false, false);
    if (nodes[i % n].state.try_claim(&was)) atomicAdd(&wins[i % n], 1ull);
  }
}

static void test_claim_exclusivity() {
  std::printf("=== try_claim exclusivity ===\n");
  const int64_t kNodes = 4096;
  Dev<Node> nodes(kNodes);
  Dev<unsigned long long> wins(kNodes);

  init_states_kernel<<<64, 256>>>(nodes.p, kNodes);
  CU(cudaDeviceSynchronize());

  hammer_claim_kernel<<<128, 256>>>(nodes.p, kNodes, wins.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto w = wins.get();
  int64_t bad = 0;
  for (int64_t i = 0; i < kNodes; ++i)
    if (w[i] != 1ull) ++bad;
  CHECK(bad == 0, "%lld of %lld nodes were claimed a number of times other "
                  "than once", (long long)bad, (long long)kNodes);
}

// ---------------------------------------------------------------------------
// 3. The queue, block-collective
// ---------------------------------------------------------------------------
//
// Drain it, then refill it, and check every node came back exactly once. This
// is also the first thing that runs the block-collective launch shape: a
// partial last wave is what would hang it.

__global__ void drain_kernel(Queue q, Node* nodes, int64_t n, int32_t* out) {
  __shared__ Queue::Scratch s;
  const uint64_t snap = q.snapshot_tail();
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    NodeState was = NodeState::make(NodeState::kEmpty, 0u, false, false);
    const int32_t got = q.pop(i < n, snap, s, [&](int32_t c) {
      return nodes[c].state.try_claim(&was);
    });
    if (i < n) out[i] = got;
  }
}

__global__ void refill_kernel(Queue q, Node* nodes, const int32_t* ids,
                              int64_t n) {
  __shared__ Queue::Scratch s;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    const int32_t node = (i < n) ? ids[i] : Queue::kNoNode;
    // Claimed nodes hold one reference; give it back so they go idle again.
    const bool owns = (node != Queue::kNoNode) && nodes[node].state.release();
    q.push(owns, node, s);
  }
}

static void test_queue_round_trip() {
  std::printf("=== reclaim queue, block-collective ===\n");
  // Deliberately not a power of two, and deliberately not a multiple of the
  // block size, so the last wave is partial.
  const int64_t kNodes = 3000;
  Dev<Node> nodes(kNodes);
  Dev<int32_t> got(kNodes);
  QStore qs(kNodes);

  init_states_kernel<<<64, 256>>>(nodes.p, kNodes);
  CU(cudaDeviceSynchronize());
  CHECK(qs.size() == kNodes, "a fresh queue holds every node, got %lld",
        (long long)qs.size());

  drain_kernel<<<32, 256>>>(qs.queue(), nodes.p, kNodes, got.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto ids = got.get();
  std::vector<int> seen(kNodes, 0);
  int64_t dry = 0, dup = 0;
  for (int32_t id : ids) {
    if (id == Queue::kNoNode) { ++dry; continue; }
    if (seen[id]++) ++dup;
  }
  CHECK(dry == 0, "%lld pops came back empty from a full queue", (long long)dry);
  CHECK(dup == 0, "%lld nodes were handed out more than once", (long long)dup);
  CHECK(qs.size() == 0, "a drained queue is empty, got %lld",
        (long long)qs.size());

  refill_kernel<<<32, 256>>>(qs.queue(), nodes.p, got.p, kNodes);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());
  CHECK(qs.size() == kNodes, "everything went back, got %lld",
        (long long)qs.size());
}

// ---------------------------------------------------------------------------
// 4. The whole bookkeeping pipeline, many steps
// ---------------------------------------------------------------------------

__global__ void fill_stub_kernel(Node* nodes, const Slot* slots,
                                 const unsigned long long* count,
                                 int64_t* rows) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const Slot s = slots[i];
    if (s.node == Map::kNoNode) continue;
    rows[s.node] = s.key;   // the row, standing in for an embedding
    nodes[s.node].state.publish();
  }
}

__global__ void verify_kernel(const Node* nodes, const int64_t* keys,
                              const int32_t* node_ids, const int64_t* rows,
                              int64_t n, unsigned long long* wrong,
                              unsigned long long* dry) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int32_t node = node_ids[i];
    if (node == Map::kNoNode) { atomicAdd(dry, 1ull); continue; }
    if (rows[node] != keys[i] || nodes[node].key != keys[i]) {
      atomicAdd(wrong, 1ull);
    }
  }
}

static void test_pipeline(int steps, int64_t capacity, int64_t per_step,
                          int64_t key_space, unsigned seed) {
  std::printf("=== pipeline: %d steps, capacity %lld, %lld keys/step out of "
              "%lld ===\n",
              steps, (long long)capacity, (long long)per_step,
              (long long)key_space);

  MapStore map_store(capacity);
  QStore qs(capacity);
  Node* nodes = map_store.nodes();

  init_states_kernel<<<64, 256>>>(nodes, capacity);
  CU(cudaDeviceSynchronize());

  Dev<int64_t> rows(capacity);
  Dev<int64_t> keys(per_step), miss_keys(per_step), miss_src(per_step);
  Dev<int32_t> node_ids(per_step);
  Dev<Slot> slots(per_step);
  Dev<unsigned long long> miss_count(1), wrong(1), dry(1);

  std::mt19937_64 rng(seed);
  // Skewed, so the same keys come back and the cache is actually exercised.
  std::geometric_distribution<int64_t> hot(0.01);

  const int grid = grid_for(per_step);
  int64_t total_miss = 0;

  for (int step = 0; step < steps; ++step) {
    std::vector<int64_t> h_keys;
    h_keys.reserve(per_step);
    std::vector<char> present(key_space, 0);
    while ((int64_t)h_keys.size() < per_step) {
      const int64_t k = hot(rng) % key_space;
      if (present[k]) continue;  // unique within a step, as the cache requires
      present[k] = 1;
      h_keys.push_back(k);
    }
    keys.put(h_keys);
    CU(cudaMemset(miss_count.p, 0, sizeof(unsigned long long)));

    lookup_pin_kernel<Map, Node><<<grid, kBlock>>>(
        map_store.map(), nodes, keys.p, per_step, node_ids.p, miss_keys.p,
        miss_src.p, miss_count.p);
    CU(cudaGetLastError());

    allocate_kernel<Queue, Map, Node><<<grid, kBlock>>>(
        qs.queue(), map_store.map(), nodes, miss_keys.p, miss_src.p,
        miss_count.p, slots.p, node_ids.p);
    CU(cudaGetLastError());

    fill_stub_kernel<<<grid, kBlock>>>(nodes, slots.p, miss_count.p, rows.p);
    CU(cudaGetLastError());

    verify_kernel<<<grid, kBlock>>>(nodes, keys.p, node_ids.p, rows.p, per_step,
                                    wrong.p, dry.p);
    CU(cudaGetLastError());

    release_kernel<Queue, Node><<<grid, kBlock>>>(qs.queue(), nodes,
                                                  node_ids.p, per_step);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    total_miss += (int64_t)miss_count.get()[0];
  }

  const auto n_wrong = wrong.get()[0];
  const auto n_dry = dry.get()[0];
  const double hit = 1.0 - double(total_miss) / double(steps * per_step);
  std::printf("  hit rate %.3f, %llu unserved, %llu wrong\n", hit, n_dry,
              n_wrong);

  CHECK(n_wrong == 0, "%llu keys resolved to a node holding something else",
        n_wrong);
  CHECK(map_store.check_bindings() == 0,
        "every node in a bucket hashes to that bucket");

  // No key may hold two nodes. check_bindings cannot see this — both nodes sit
  // in the right bucket — so it is checked here.
  const auto h_nodes = map_store.download_nodes();
  const auto h_heads = map_store.download_heads();
  std::map<int64_t, int> owner;
  int64_t dupes = 0;
  for (int64_t b = 0; b < map_store.num_buckets(); ++b) {
    for (int32_t cur = h_heads[b]; cur != Map::kNoNode;
         cur = h_nodes[cur].next) {
      if (!owner.emplace(h_nodes[cur].key, cur).second) ++dupes;
    }
  }
  CHECK(dupes == 0, "%lld keys are bound to more than one node",
        (long long)dupes);

  // At rest every node is idle, and every idle node is queued exactly once.
  int64_t queued = 0, referenced = 0;
  for (int64_t i = 0; i < capacity; ++i) {
    const NodeState s = h_nodes[i].state.get();
    if (s.has_refs()) ++referenced;
    if (s.in_ring()) ++queued;
  }
  CHECK(referenced == 0, "%lld nodes still hold a reference after every "
                         "release", (long long)referenced);
  std::printf("  %lld nodes flagged queued, queue holds %lld\n",
              (long long)queued, (long long)qs.size());
  CHECK(queued == qs.size(),
        "the in_queue flags and the queue itself disagree: %lld vs %lld",
        (long long)queued, (long long)qs.size());
}

int main() {
  int dev = 0;
  CU(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CU(cudaGetDeviceProperties(&prop, dev));
  std::printf("device: %s, sm_%d%d, %d SMs\n\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  test_state_contention();
  test_claim_exclusivity();
  test_queue_round_trip();
  // Capacity below the working set on purpose: eviction has to actually happen.
  test_pipeline(/*steps=*/200, /*capacity=*/2048, /*per_step=*/512,
                /*key_space=*/8192, /*seed=*/1234);
  test_pipeline(/*steps=*/50, /*capacity=*/600, /*per_step=*/500,
                /*key_space=*/4096, /*seed=*/99);

  if (failures == 0) {
    std::printf("\nall GPU checks passed\n");
    return 0;
  }
  std::printf("\n%d check(s) failed\n", failures);
  return 1;
}
