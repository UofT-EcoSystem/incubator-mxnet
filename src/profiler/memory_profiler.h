#pragma once

#include <fstream>

#include <mxnet/memory_profiling.h>

#if MXNET_USE_MEMORY_PROFILER

namespace mxnet {
  namespace profiler {

class MemoryProfiler {
 private:
  bool _use_memory_profiler;
  std::ofstream _fout;
  std::ofstream _ferr;
 public:
  MemoryProfiler();
  static MemoryProfiler * Get();
  void addEntry(
    const std::size_t alloc_size,
    const std::string & tag);
};

  } // namespace profiler
} // namespace mxnet

#endif // MXNET_USE_MEMORY_PROFILER