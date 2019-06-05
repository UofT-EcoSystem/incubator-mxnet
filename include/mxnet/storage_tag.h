#pragma once

#include <string>

namespace mxnet {

namespace {
/// \brief Given a path, extract the filename (i.e., remove directory).
///        This is equivalent of the following bash command:
/// \code{.sh}
/// basename ${path}
/// \endcode
inline std::string __extract_fname(const std::string& path) {
  std::size_t last_dir_pos = path.find_last_of("/\\");
  if (last_dir_pos == std::string::npos) {
    last_dir_pos = 0;
  }
  return path.substr(last_dir_pos + 1);
}
}  // namespace

/** Default Storage Tag 
 *
 *  If we are not using this default Storage tag, then for each operator
 *   that uses `Storage` and/or `Resource` allocations, we need to make
 *   **intrusive** changes by tagging those allocations inside every
 *   operator, which is apparently a very tedious and non-scalable solution.
 *
 *  Note that since the built-in functions `__builtin_*` are ONLY supported
 *    under GCC, the check on `__GNUG__`
 *    (which is equivalent to `__GNUC__ && __cplusplus`) is added.
 */
#ifdef __GNUG__  // If complied with GCC
#define MXNET_DEFAULT_STORAGE_TAG(tag) std::string(tag) \
    + ":" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__) \
    + ":" + __extract_fname(__builtin_FILE()) \
    + "+" +  std::to_string(__builtin_LINE()) \
    + ":" + __builtin_FUNCTION()
#else  // !__GNUG__
#define MXNET_DEFAULT_STORAGE_TAG(tag) std::string(tag) \
    + ":" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__)
#endif  // __GNUG__

}  // namespace mxnet
