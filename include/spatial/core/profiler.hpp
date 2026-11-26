#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace spatial {

struct MemoryStats {
    double rss_mb       = 0.0;  // Resident set size
    double footprint_mb = 0.0;  // macOS phys_footprint (actual physical memory pressure)
};

MemoryStats get_memory_stats();

struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start = Clock::now();

    void reset() { start = Clock::now(); }
    double elapsed_ms() const {
        auto dt = Clock::now() - start;
        return std::chrono::duration<double, std::milli>(dt).count();
    }
};

struct ProfileEntry {
    std::string name;
    double      ms = 0.0;
};

class Profiler {
public:
    void record(const std::string& name, double ms) {
        entries_.push_back({name, ms});
    }
    void clear() { entries_.clear(); }
    void print_summary() const;

private:
    std::vector<ProfileEntry> entries_;
};

}  // namespace spatial
