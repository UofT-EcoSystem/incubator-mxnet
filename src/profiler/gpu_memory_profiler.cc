#include <dmlc/logging.h>

#include "./gpu_memory_profiler.h"

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
#define MB (1024.0 * 1024.0)
	_fout << tag << "," << alloc_size / MB << std::endl;

	if (tag == "untagged" || tag == "<unk>" || 
	    tag == "" || 
	    tag == "parameter:bertencoder0_transformer11_bertpositionwiseffn0_bertlayernorm0_gamma")
	{
		_ferr << dmlc::StackTrace() << std::endl;
		_ferr << "[gpu_memory_profiler:info] " << 
		         "Allocating " << alloc_size / MB << " "
		         "with Context Tag " << tag << std::endl;
	}
#undef MB
}

	} // namespace profiler
} // namespace mxnet

#endif // MXNET_USE_MEMORY_PROFILER
