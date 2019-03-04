#pragma once

#include <string>
#include <fstream>

#include <mxnet/base.h>

#if MXNET_USE_MEMORY_PROFILER
namespace mxnet {
	namespace profiler {

class GpuMemoryProfiler
{
private:
	std::ofstream _fout;
	std::ofstream _ferr;
public:
	 GpuMemoryProfiler();
	~GpuMemoryProfiler();

	void addEntry(const std::string & tag,
	              const std::size_t   alloc_size);
};

	} // namespace profiler
} // namespace mxnet
#endif // MXNET_USE_MEMORY_PROFILER
