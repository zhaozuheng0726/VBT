#include "scientific_decode.h"
#include "vbt_file.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace vbt::render;

namespace {

void usage()
{
    std::cout
        << "Usage: vbt_export_scientific_frame --input-vbt <file.vbtp> --frame <index> --output-raw <frame.raw>\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path input;
    std::filesystem::path output;
    uint32_t frame = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) input = argv[++i];
        else if (arg == "--frame" && i + 1 < argc) frame = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--output-raw" && i + 1 < argc) output = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        }
    }

    if (input.empty() || output.empty()) {
        usage();
        return 1;
    }

    VbtFile file;
    std::string error;
    if (!loadVbtFile(input, file, error)) {
        std::cerr << error << "\n";
        return 2;
    }
    if (frame >= file.header.frames) {
        std::cerr << "Frame index out of range.\n";
        return 3;
    }

    const auto reconstructed = reconstructScientificFrameCpu(file, frame);
    std::ofstream out(output, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output raw: " << output.string() << "\n";
        return 4;
    }
    out.write(reinterpret_cast<const char*>(reconstructed.data()),
              static_cast<std::streamsize>(reconstructed.size() * sizeof(float)));
    if (!out) {
        std::cerr << "Failed to write output raw.\n";
        return 5;
    }

    std::cout << "Exported frame " << frame << " to " << output.string() << "\n";
    return 0;
}
