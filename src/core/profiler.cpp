#include "spatial/core/profiler.hpp"

#include <cstdio>
#include <unordered_map>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

namespace spatial {

MemoryStats get_memory_stats() {
    MemoryStats stats;
#ifdef __APPLE__
    task_vm_info_data_t vm_info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &count) == KERN_SUCCESS) {
        stats.rss_mb       = static_cast<double>(vm_info.resident_size) / (1024.0 * 1024.0);
        stats.footprint_mb = static_cast<double>(vm_info.phys_footprint) / (1024.0 * 1024.0);
    }
#endif
    return stats;
}

void Profiler::print_summary() const {
    if (entries_.empty()) return;
    struct Agg { double total = 0.0; int count = 0; };
    std::unordered_map<std::string, Agg> agg;
    std::vector<std::string> order;
    double grand_total = 0.0;
    for (const auto& e : entries_) {
        if (agg.find(e.name) == agg.end()) order.push_back(e.name);
        agg[e.name].total += e.ms;
        agg[e.name].count += 1;
        grand_total += e.ms;
    }
    std::printf("\n--- Spatial Profiler Summary (total=%.2f ms) ---\n", grand_total);
    for (const auto& name : order) {
        const auto& a = agg[name];
        double pct = grand_total > 0.0 ? (100.0 * a.total / grand_total) : 0.0;
        std::printf("  %-28s %8.2f ms (%5.1f%%)  [%dx, avg %.2f ms]\n",
                    name.c_str(), a.total, pct, a.count, a.total / a.count);
    }
    auto mem = get_memory_stats();
    std::printf("  Memory: phys_footprint=%.1f MB | RSS=%.1f MB\n",
                mem.footprint_mb, mem.rss_mb);
}

}  // namespace spatial
