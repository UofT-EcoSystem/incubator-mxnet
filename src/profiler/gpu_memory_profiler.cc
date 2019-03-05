#include <dmlc/logging.h>

#include "./gpu_memory_profiler.h"

inline std::string _StackTrace() {
  using std::string;
  std::ostringstream stacktrace_os;
  const int MAX_STACK_SIZE = 32;
  void *stack[MAX_STACK_SIZE];
  int nframes = backtrace(stack, MAX_STACK_SIZE);
  stacktrace_os << "Stack trace returned " << nframes << " entries:" << std::endl;
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int frameno = 0; frameno < nframes; ++frameno) {
      string msg = dmlc::Demangle(msgs[frameno]);
      stacktrace_os << "[bt] (" << frameno << ") " << msg << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}

#if MXNET_USE_MEMORY_PROFILER
mxnet::profiler::GpuMemoryProfiler g_gpu_memory_profiler;

namespace mxnet {
	namespace profiler {

GpuMemoryProfiler:: GpuMemoryProfiler()
{
	const char * fout_name = getenv("MXNET_PROFILER_FOUT_NAME");
	const char * ferr_name = getenv("MXNET_PROFILER_FERR_NAME");

	if (fout_name == nullptr)
	{
		_fout.open("/tmp/mxnet_memory_profiler_output.csv");
	}
	else { _fout.open(fout_name); }

	if (ferr_name == nullptr)
	{
		_ferr.open("/tmp/mxnet_memory_profiler_output.log");
	}
	else { _ferr.open(ferr_name); }
}

GpuMemoryProfiler::~GpuMemoryProfiler() { _fout.close(); _ferr.close(); }

void GpuMemoryProfiler::addEntry(const std::string & tag,
                                 const std::size_t   alloc_size)
{
#define MB 1024 * 1024
	_fout << tag << "," << alloc_size << std::endl;
	std::cout << "[gpu_memory_profiler:info] " << 
		"Allocating " << alloc_size / MB << " "
		"with Context Tag " << tag << std::endl;

	if (tag == "untagged")
	{
		// unknown tags from the Python end
		    _ferr << _StackTrace() << std::endl;
		std::cerr << _StackTrace() << std::endl;
	}
	if (tag == "<unk>")
	{
		// unknown tags from the C++ end
		    _ferr << _StackTrace() << std::endl;
		std::cerr << _StackTrace() << std::endl;
	}
	if (tag == "_zeros")
	{
		    _ferr << _StackTrace() << std::endl;
		std::cerr << _StackTrace() << std::endl;
	}
	if (tag == "broadcast_lesser")
	{
		    _ferr << _StackTrace() << std::endl;
		std::cerr << _StackTrace() << std::endl;
	}
#undef MB
}

	} // namespace profiler
} // namespace mxnet

#endif // MXNET_USE_MEMORY_PROFILER