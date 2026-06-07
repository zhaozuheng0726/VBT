#include "vbt_cycles_blob.h"
#include "vbt_cycles_loader.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Probe {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

void print_usage()
{
    std::cout
        << "Usage:\n"
        << "  vbt_sampler_cpu_probe --input-vbt <file.vbtp> [--frame N]\n"
        << "                        [--probe x,y,z] ...\n\n"
        << "Coordinates are VBT index-space coordinates, not Blender world space.\n";
}

bool parse_probe(const std::string &text, Probe &probe)
{
    const size_t a = text.find(',');
    if (a == std::string::npos) {
        return false;
    }
    const size_t b = text.find(',', a + 1);
    if (b == std::string::npos) {
        return false;
    }
    probe.x = std::stof(text.substr(0, a));
    probe.y = std::stof(text.substr(a + 1, b - a - 1));
    probe.z = std::stof(text.substr(b + 1));
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    std::filesystem::path input;
    int frame = 0;
    std::vector<Probe> probes;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) {
            input = argv[++i];
        }
        else if (arg == "--frame" && i + 1 < argc) {
            frame = std::stoi(argv[++i]);
        }
        else if (arg == "--probe" && i + 1 < argc) {
            Probe probe;
            if (!parse_probe(argv[++i], probe)) {
                std::cerr << "Invalid --probe value. Expected x,y,z.\n";
                return 2;
            }
            probes.push_back(probe);
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 2;
        }
    }

    if (input.empty()) {
        print_usage();
        return 2;
    }

    vbt_cycles_host::File file;
    std::string error;
    if (!vbt_cycles_host::load_pack4(input, file, error)) {
        std::cerr << error << "\n";
        return 3;
    }
    if (!vbt_cycles_host::is_density_payload(file)) {
        std::cerr << "Warning: payload profile is not density. The temporal render sampler is intended for smoke/fire density VBT files.\n";
    }

    if (probes.empty()) {
        probes.push_back({0.0f, 0.0f, 0.0f});
        probes.push_back({static_cast<float>(file.header.width) * 0.25f,
                          static_cast<float>(file.header.height) * 0.50f,
                          static_cast<float>(file.header.depth) * 0.50f});
        probes.push_back({static_cast<float>(file.header.width) * 0.50f,
                          static_cast<float>(file.header.height) * 0.50f,
                          static_cast<float>(file.header.depth) * 0.50f});
        probes.push_back({static_cast<float>(file.header.width > 0 ? file.header.width - 1u : 0u),
                          static_cast<float>(file.header.height > 0 ? file.header.height - 1u : 0u),
                          static_cast<float>(file.header.depth > 0 ? file.header.depth - 1u : 0u)});
    }

    std::vector<uint8_t> blob;
    if (!vbt_cycles_host::pack_blob(file, frame, blob, error)) {
        std::cerr << error << "\n";
        return 4;
    }

    vbt_cycles::PayloadView view;
    if (!vbt_cycles_host::blob_payload_view(blob, view, error)) {
        std::cerr << error << "\n";
        return 5;
    }

    std::cout << "x,y,z,frame,blob_bytes,density\n";
    std::cout << std::setprecision(9);
    for (const Probe &probe : probes) {
        const float density = vbt_cycles::sample_density_index(view, probe.x, probe.y, probe.z);
        std::cout << probe.x << "," << probe.y << "," << probe.z << ","
                  << view.header.frame_index << "," << blob.size() << "," << density << "\n";
    }

    return 0;
}
