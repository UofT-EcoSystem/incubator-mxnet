/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "./storage_profiler.h"

#include <dmlc/json.h>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#if MXNET_USE_NVML
#include <nvml.h>
#endif  // MXNET_USE_NVML

#include "./profiler.h"
#include "../common/utils.h"
#include "../common/cuda/utils.h"


namespace dmlc {
namespace json {

template<typename T>
struct Handler<std::shared_ptr<T> >  {
  static void Write(JSONWriter* const writer,
                    const std::shared_ptr<T>& ptr) {
    writer->Write(*ptr);
  }
};

}  // namespace json
}  // namespace dmlc


namespace mxnet {
namespace profiler {

namespace {

struct GpuMemProfileNode {
  std::string name;

  GpuMemProfileNode(const std::string& name_) : name(name_) {}
  virtual ~GpuMemProfileNode() {}
  virtual void Save(dmlc::JSONWriter* const writer) const {
    writer->WriteObjectKeyValue("name", name);
  }
};  // struct GpuMemProfileNode

using GpuMemProfileNodePtr = std::shared_ptr<GpuMemProfileNode>;

struct GpuMemProfileAttributeNode : GpuMemProfileNode {
  std::size_t size;

  GpuMemProfileAttributeNode(const std::string& name_,
                             const std::size_t size_)
      : GpuMemProfileNode(name_), size(size_)
  {}
  virtual void Save(dmlc::JSONWriter* const writer) const override {
    writer->BeginObject(false);
    GpuMemProfileNode::Save(writer);
    writer->WriteObjectKeyValue("value", size);
    writer->EndObject();
  }
};  // struct GpuMemProfileAttributeNode

struct GpuMemProfileScopeNode : GpuMemProfileNode {
  std::vector<GpuMemProfileNodePtr> children;

  GpuMemProfileScopeNode(const std::string& name_) : GpuMemProfileNode(name_)
  {}
  virtual ~GpuMemProfileScopeNode() {}
  /**!
   * \brief Insert allocation entry (name, size) pair into the tree.
   */
  void Insert(const std::string& alloc_entry_name,
              const size_t size) {
    // try to find the profiler scope separator in the allocation entry name
    size_t profiler_scope_seperator_pos = alloc_entry_name.find(":");
    if (profiler_scope_seperator_pos == std::string::npos) {
      // If the separator is NOT found, then this implies that the entry is a
      // leaf node and we insert it directly into children.
      children.emplace_back(
          new GpuMemProfileAttributeNode(alloc_entry_name, size));
    } else {
      // retrieve the leading profiler scope and the remainder from the name of
      // the allocation entry
      const std::string
            alloc_entry_profiler_scope
              = alloc_entry_name.substr(0, profiler_scope_seperator_pos),
            alloc_entry_remainder
              = alloc_entry_name.substr(profiler_scope_seperator_pos + 1);

      // traverse through the children to find the leading profiler scope
      for (GpuMemProfileNodePtr& child : children) {
        if (GpuMemProfileScopeNode* const profiler_scope_node =
            dynamic_cast<GpuMemProfileScopeNode*>(child.get())) {
          if (child->name == alloc_entry_profiler_scope) {
            profiler_scope_node->Insert(alloc_entry_remainder, size);
            return;
          }  // if (child->name == alloc_entry_profiler_scope)
        }  // profiler_scope_node
           //   = dyn_cast<ProfilerScopeNode>(child)
      }  // for (child ∈ children)
      // append the profiler scope at the end if it has NOT been found
      children.emplace_back(
          new GpuMemProfileScopeNode(alloc_entry_profiler_scope));
      GpuMemProfileScopeNode* const profiler_scope_node
          = dynamic_cast<GpuMemProfileScopeNode*>(children.back().get());
      profiler_scope_node->Insert(alloc_entry_remainder, size);
    }  // if (profiler_scope_seperator_pos == std::string::npos)
  }
  virtual void Save(dmlc::JSONWriter* const writer) const override {
    writer->BeginObject();
    GpuMemProfileNode::Save(writer);
    writer->WriteObjectKeyValue("children", children);
    writer->EndObject();
  }
};  // struct GpuMemProfileScopeNode

/*!
 * \brief The @c GpuMemProfileTree is the data structure for storing the GPU
 *        memory profile information in a tree-like structure. This is needed
 *        when we are dumping the profile in JSON format.
 */
struct GpuMemProfileTree {
  GpuMemProfileNodePtr root;

  GpuMemProfileTree(const int dev_id) {
    root = std::make_shared<GpuMemProfileScopeNode>(
        "gpu_memory_profile_dev" + std::to_string(dev_id));
  }
  void Save(dmlc::JSONWriter* const writer) const {
    writer->Write(root);
  }
  /**!
   * \brief Insert allocation entry (name, size) pair into the tree.
   */
  void Insert(const std::string& alloc_entry_name,
              const size_t size) {
    std::dynamic_pointer_cast<GpuMemProfileScopeNode>(root)->Insert(alloc_entry_name, size);
  }
};  // struct GpuMemProfileTree

/*! JSON Graph for Serialization
 */
class GpuMemProfileJSONGraph {
 private:
  // dev ID -> GpuMemprofileTree
  std::map<int, GpuMemProfileTree> trees_;
 public:
  GpuMemProfileTree& operator[](const int dev_id) {
    if (trees_.find(dev_id) == trees_.end()) {
      return trees_.emplace(dev_id, GpuMemProfileTree(dev_id)).first->second;
    }
    return trees_.at(dev_id);
  }
  void Save(dmlc::JSONWriter* const writer) const {
    writer->BeginObject();
    writer->BeginArray(true);
    for (const std::pair<int, GpuMemProfileTree>& dev_id_tree_pair
         : trees_) {
      writer->WriteArrayItem(dev_id_tree_pair.second);
    }
    writer->EndArray();
    writer->EndObject();
  }
};  // struct GPUMemProfileJSONGraph

}   // namespace anonymous

#if MXNET_USE_CUDA
GpuDeviceStorageProfiler* GpuDeviceStorageProfiler::Get() {
  static std::mutex mtx;
  static std::shared_ptr<GpuDeviceStorageProfiler> gpu_dev_storage_profiler = nullptr;
  std::unique_lock<std::mutex> lk(mtx);
  if (!gpu_dev_storage_profiler) {
    gpu_dev_storage_profiler = std::make_shared<GpuDeviceStorageProfiler>();
  }
  return gpu_dev_storage_profiler.get();
}

void GpuDeviceStorageProfiler::DumpProfile() const {
  size_t current_pid = common::current_process_id();
  std::ofstream csv_fout((filename_prefix_ + "-pid_" + std::to_string(current_pid)
                          + ".csv").c_str());
  GpuMemProfileJSONGraph jgraph;
  if (!csv_fout.is_open()) {
    return;
  }
  struct AllocEntryDumpFmt {
    size_t requested_size;
    int dev_id;
    size_t actual_size;
    bool reuse;
  };
  // order the GPU memory allocation entries by their attribute name
  std::multimap<std::string, AllocEntryDumpFmt> gpu_mem_ordered_alloc_entries;
  // map the GPU device ID to the total amount of allocations
  std::unordered_map<int, size_t> gpu_dev_id_total_alloc_map;
  std::regex gluon_param_regex("([0-9a-fA-F]{8})_([0-9a-fA-F]{4})_"
                               "([0-9a-fA-F]{4})_([0-9a-fA-F]{4})_"
                               "([0-9a-fA-F]{12})_([^ ]*)");

  for (const std::pair<void *const, AllocEntry>& alloc_entry :
       gpu_mem_alloc_entries_) {
    std::string alloc_entry_name
        = std::regex_replace(alloc_entry.second.name, gluon_param_regex, "$6");
    if (alloc_entry_name == "") {
      // If the entry name becomes none after the regex replacement, we revert
      // back to the original.
      alloc_entry_name = alloc_entry.second.name;
    }
    gpu_mem_ordered_alloc_entries.emplace(
        alloc_entry.second.profiler_scope + alloc_entry_name,
        AllocEntryDumpFmt{
          alloc_entry.second.requested_size,
          alloc_entry.second.dev_id,
          alloc_entry.second.actual_size,
          alloc_entry.second.reuse});
    gpu_dev_id_total_alloc_map[alloc_entry.second.dev_id] = 0;
  }
  csv_fout << "\"Attribute Name\",\"Requested Size\","
              "\"Device\",\"Actual Size\",\"Reuse?\"" << std::endl;
  for (const std::pair<const std::string, AllocEntryDumpFmt>& alloc_entry :
       gpu_mem_ordered_alloc_entries) {
    jgraph[alloc_entry.second.dev_id].Insert(
        alloc_entry.first,
        alloc_entry.second.actual_size);
    csv_fout << "\"" << alloc_entry.first << "\","
             << "\"" << alloc_entry.second.requested_size << "\","
             << "\"" << alloc_entry.second.dev_id << "\","
             << "\"" << alloc_entry.second.actual_size << "\","
             << "\"" << alloc_entry.second.reuse << "\"" << std::endl;
    gpu_dev_id_total_alloc_map[alloc_entry.second.dev_id] +=
        alloc_entry.second.actual_size;
  }
#if MXNET_USE_NVML
  // If NVML has been enabled, add amend term to the GPU memory profile.
  nvmlDevice_t nvml_device;

  NVML_CALL(nvmlInit());
  for (std::pair<const int, size_t>& dev_id_total_alloc_pair :
       gpu_dev_id_total_alloc_map) {
    unsigned info_count = 0;
    std::vector<nvmlProcessInfo_t> infos(info_count);

    NVML_CALL(nvmlDeviceGetHandleByIndex(dev_id_total_alloc_pair.first, &nvml_device));
    // The first call to `nvmlDeviceGetComputeRunningProcesses` is to set the
    // size of info. Since `NVML_ERROR_INSUFFICIENT_SIZE` will always be
    // returned, we do not wrap the function call with `NVML_CALL`.
    nvmlDeviceGetComputeRunningProcesses(nvml_device, &info_count, infos.data());
    infos = std::vector<nvmlProcessInfo_t>(info_count);
    NVML_CALL(nvmlDeviceGetComputeRunningProcesses(nvml_device, &info_count, infos.data()));

    bool amend_made = false;

    for (unsigned i = 0; i < info_count; ++i) {
      if (current_pid == infos[i].pid) {
        amend_made = true;
        const size_t nvml_amend_size
            = infos[i].usedGpuMemory - dev_id_total_alloc_pair.second;
        jgraph[dev_id_total_alloc_pair.first].Insert(
            "nvml_amend", nvml_amend_size);
        csv_fout << "\"" << "nvml_amend" << "\","
                 << "\"" << nvml_amend_size << "\","
                 << "\"" << dev_id_total_alloc_pair.first << "\","
                 << "\"" << nvml_amend_size << "\","
                 << "\"0\"" << std::endl;
        break;
      }
    }
    if (!amend_made) {
      LOG(INFO) << "NVML is unable to make amendment to the GPU memory profile "
                   "since it is unable to locate the current process ID. "
                   "Are you working in Docker without setting --pid=host?";
    }
  }  // for (dev_id_total_alloc_pair : gpu_dev_id_total_alloc_map)
#endif  // MXNET_USE_NVML
  std::ofstream json_fout((filename_prefix_ + ".json").c_str());
  dmlc::JSONWriter writer(&json_fout);
  jgraph.Save(&writer);
}
#endif  // MXNET_USE_CUDA

}  // namespace profiler
}  // namespace mxnet
