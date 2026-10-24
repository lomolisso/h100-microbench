/*
 * gpc_map.hpp — reads the smid → GPC map produced by experiment 03 (latest run_id of
 * ../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv) and picks representative SMs.
 *   load_gpc_map()       gpc[smid] (−1 for an SM missing from the map); exits if the file is missing.
 *   first_sm_of_gpc(m,g) smallest smid of GPC g (03 labels GPCs by their smallest smid, so G g ↔ smid 2g here).
 */
#pragma once
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "host.hpp"

inline std::vector<int> load_gpc_map(int sm_count, const char* path = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv") {
    std::ifstream f(path);
    if (!f) hb::die(std::string("cannot open the 03 map ") + path);
    auto split = [](const std::string& l) {
        std::vector<std::string> v;
        std::stringstream ss(l);
        for (std::string c; std::getline(ss, c, ',');) v.push_back(c);
        return v;
    };
    std::string line;
    std::getline(f, line);
    const auto h = split(line);
    int c_run = -1, c_smid = -1, c_gpc = -1;
    for (int i = 0; i < (int)h.size(); ++i) {
        if (h[i] == "run_id") c_run = i;
        if (h[i] == "smid") c_smid = i;
        if (h[i] == "gpc") c_gpc = i;
    }
    if (c_run < 0 || c_smid < 0 || c_gpc < 0) hb::die("03 map lacks run_id/smid/gpc columns");
    std::vector<std::vector<std::string>> rows;
    std::string latest;
    while (std::getline(f, line)) {
        auto r = split(line);
        if ((int)r.size() <= c_gpc) continue;
        if (r[c_run] > latest) latest = r[c_run];  // ISO timestamps with one UTC offset sort lexicographically
        rows.push_back(std::move(r));
    }
    std::vector<int> gpc(sm_count, -1);
    for (const auto& r : rows)
        if (r[c_run] == latest) {
            const int s = std::stoi(r[c_smid]);
            if (s >= 0 && s < sm_count) gpc[s] = std::stoi(r[c_gpc]);
        }
    for (int s = 0; s < sm_count; ++s)
        if (gpc[s] < 0) hb::die("03 map has no GPC for smid " + std::to_string(s));
    return gpc;
}

inline int first_sm_of_gpc(const std::vector<int>& gpc, int g) {
    for (int s = 0; s < (int)gpc.size(); ++s)
        if (gpc[s] == g) return s;
    hb::die("no SM in GPC " + std::to_string(g));
}
