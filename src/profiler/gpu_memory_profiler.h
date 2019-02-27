#pragma once

#include <string>
#include <utility>
#include <cstdlib>
#include <fstream>

#define MXNET_USE_MEMORY_PROFILER 1

#if MXNET_USE_MEMORY_PROFILER
namespace mxnet {
	namespace profiler {
class GpuMemoryProfiler
{
private:
	std::ofstream _fout;
public:
	 GpuMemoryProfiler()
	{
		const char * ofname = getenv("MXNET_PROFILER_OFNAME");

		if (ofname == nullptr)
		{
			_fout.open("/tmp/mxnet_memory_profiler_output.csv");
		}
		else { _fout.open(ofname); }
	}
	~GpuMemoryProfiler() { _fout.close(); }
	
	void addEntry(const std::string & tag,
	              const std::size_t   alloc_size)
	{
		_fout << tag << "," << alloc_size << std::endl;;
	}
};
	} // namespace profiler
} // namespace mxnet
#endif // MXNET_USE_MEMORY_PROFILER
