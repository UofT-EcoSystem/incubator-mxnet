#pragma once

#include <mxnet/storage_tag.h>
#include <string>
#include <fstream>

namespace mxnet {
namespace profiler {

class GpuMemoryProfiler {
 private:
  bool _enabled;  ///< Even if compiled with the GPU memory profiler enabled,
                  ///    frontend programmers can still disable it using
                  ///    the environment variable `MXNET_USE_GPU_MEMORY_PROFILER`.
  std::ofstream _csv_fout;  ///< `_csv_fout` is used for storing
                            ///    GPU memory allocation entries in CSV format.
                            ///  The file location can be set using environment variable
                            ///    `MXNET_GPU_MEMORY_PROFILER_CSV_FNAME`
  std::ofstream _log_fout;  ///< `_log_fout` is used for logging
                            ///    unknown entries and their callstack.
                            ///  The file location can be set using environment variable
                            ///    `MXNET_GPU_MEMORY_PROFILER_LOG_FNAME`
 public:
  /// \brief Get the static instance of the GPU memory profiler.
  static GpuMemoryProfiler * Get();
  GpuMemoryProfiler();
  /// \brief Record a GPU memory allocation entry in the memory profiler.
  void addEntry(const std::size_t alloc_size,
      const std::string & tag);
};

}  // namespace profiler
}  // namespace mxnet
