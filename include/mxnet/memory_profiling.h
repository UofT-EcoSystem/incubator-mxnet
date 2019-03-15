#pragma once

#include <string>

#define MXNET_USE_MEMORY_PROFILER 1

#if MXNET_USE_MEMORY_PROFILER

static inline
std::string _extract_fname(const std::string & fname) {
	return fname.substr(0, fname.find_last_of('.'));
}

#define DEFAULT_MEMORY_TAG(tag) std::string(tag) \
    + ":" + _extract_fname(__FILE__) \
    + ":" + _extract_fname(__builtin_FILE()) \
    + ":" + __builtin_FUNCTION() \
    + ":" + std::to_string(__builtin_LINE())

#endif // MXNET_USE_MEMORY_PROFILER