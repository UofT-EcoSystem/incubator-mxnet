#include <cstdlib>

#include <dmlc/logging.h>

#include "./memory_profiler.h"

#if MXNET_USE_MEMORY_PROFILER

namespace mxnet {
  namespace profiler {

MemoryProfiler::MemoryProfiler() {
  const char * fout_fname = getenv("MXNET_MEMORY_PROFILER_FOUT_FNAME");
  const char * ferr_fname = getenv("MXNET_MEMORY_PROFILER_FERR_FNAME");

  if (fout_fname == nullptr) {
    _fout.open("/tmp/mxnet_memory_profiler_output.csv");
  } else {
    _fout.open(fout_fname);
  }
  if (ferr_fname == nullptr) {
    _ferr.open("/tmp/mxnet_memory_profiler_output.log");
  } else {
    _ferr.open(ferr_fname);
  }
}

MemoryProfiler * 
MemoryProfiler::Get() {
  static MemoryProfiler s_memory_profiler;
  return &s_memory_profiler;
}

void 
MemoryProfiler::addEntry(
  const std::size_t alloc_size,
  const std::string & tag) {
#define MiB (1024.0 * 1024.0)
  _fout << tag << "," << alloc_size / MiB << std::endl;

  if (tag.find("unknown") != std::string::npos) {
    const int MAX_STACK_SIZE = 10;
    void * stack [MAX_STACK_SIZE];

    int nframes = backtrace(stack, MAX_STACK_SIZE);
    _ferr << "\n\n" << "Stack trace returned " << nframes << " entries:" << std::endl;
    char ** msgs = backtrace_symbols(stack, nframes);
    if (msgs != nullptr) {
      for (int i = 0; i < nframes; ++i) {
        _ferr << "[bt] (" << i << ") " << msgs[i] << std::endl;
      }
    }
    _ferr << "[memory_profiler::warning] "
          << "Allocating " << alloc_size / MiB << " "
          << "with unknown Memory Tag " << tag << "." << std::endl;
  }
#undef MiB
}

  } // namespace profiler
} // namespace mxnet

#endif // MXNET_USE_MEMORY_PROFILER