#include "./gpu_memory_profiler.h"

#if MXNET_USE_MEMORY_PROFILER
mxnet::profiler::GpuMemoryProfiler g_gpu_memory_profiler;
#endif // MXNET_USE_MEMORY_PROFILER