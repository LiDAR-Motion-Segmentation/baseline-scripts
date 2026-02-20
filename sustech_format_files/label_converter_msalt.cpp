/**
 * @file label_converter_msalt.cpp
 * @brief Batch converts MSALT 3D bounding box labels to SUSTechPOINTS coordinate system.
 * * Compile with: g++ -std=c++17 -O3 -I. label_converter_msalt.cpp -o label_converter_msalt
 * use --reverse flag for reverse operation 
 */

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

class LabelConverter {
private:
    std::string input_dir_;
    std::string output_dir_;
    bool reverse_mode_;

    // 90 degrees in radians
    const double PI_OVER_2 = 1.5707963267948966;

public:
    LabelConverter(const std::string& input_dir, const std::string& output_dir, bool reverse_mode = false)
    : input_dir_(input_dir), output_dir_(output_dir), reverse_mode_(reverse_mode) {}

    bool processLabels() const {
        if (!fs::exists(input_dir_) || !fs::is_directory(input_dir_)) {
            std::cerr << "[ERROR] Input directory does not exist: " << input_dir_ << "\n";
            return false;
        }

        if (!fs::exists(output_dir_)) {
            fs::create_directories(output_dir_);
            std::cout << "[INFO] Created output directory: " << output_dir_ << "\n";
        }

        int processed_files = 0;

        for (const auto& entry : fs::directory_iterator(input_dir_)) {
            if (entry.path().extension() == ".json") {
                fs::path input_path = entry.path();
                fs::path output_path = fs::path(output_dir_) / input_path.filename();

                if (convertSingleFile(input_path, output_path)) {
                    processed_files++;
                }
            }
        }

        std::string mode_str = reverse_mode_ ? "SUSTechPOINTS -> MSALT" : "MSALT -> SUSTechPOINTS";
        std::cout << "[INFO] Mode: " << mode_str << "\n";
        std::cout << "[INFO] Successfully converted " << processed_files << " label files.\n";
        return true;
    }

private:
    bool convertSingleFile(const fs::path& in_path, const fs::path& out_path) const {
        try {
            std::ifstream in_file(in_path);
            if (!in_file.is_open()) {
                std::cerr << "[WARNING] Could not open file: " << in_path << "\n";
                return false;
            }

            json labels;
            in_file >> labels;
            in_file.close();

            for (auto& box : labels) {
                if (box.contains("psr")) {
                    // DO NOT TOUCH POSITIONS! They were already correct in raw MSALT.
                    // We just rewrite them exactly as they are.
                    double old_x = box["psr"]["position"]["x"];
                    double old_y = box["psr"]["position"]["y"];
                    double old_z = box["psr"]["position"]["z"];
                    
                    double old_rot_z = box["psr"]["rotation"]["z"];

                    if (!reverse_mode_) {
                        // Forward: MSALT -> SUSTechPOINTS
                        // Lock positions in place
                        box["psr"]["position"]["x"] = old_x;
                        box["psr"]["position"]["y"] = old_y;
                        box["psr"]["position"]["z"] = old_z; 

                        // Only rotate the box itself by 90 degrees (Yaw)
                        box["psr"]["rotation"]["z"] = old_rot_z + PI_OVER_2;
                    } else {
                        // Reverse: SUSTechPOINTS -> MSALT 
                        box["psr"]["position"]["x"] = old_x;
                        box["psr"]["position"]["y"] = old_y;
                        box["psr"]["position"]["z"] = old_z;

                        // Reverse the rotation
                        box["psr"]["rotation"]["z"] = old_rot_z - PI_OVER_2;
                    }
                }
            }

            std::ofstream out_file(out_path);
            out_file << std::setw(2) << labels << std::endl;
            out_file.close();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed processing " << in_path.filename() << ": " << e.what() << "\n";
            return false;
        }
    }
};

int main(int argc, char* argv[]) {
    bool reverse_mode = false;

    // Check for the optional --reverse flag
    if (argc == 4 && std::string(argv[3]) == "--reverse") {
        reverse_mode = true;
    } else if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_labels_dir> <output_labels_dir> [--reverse]\n";
        std::cerr << "Example (Forward): " << argv[0] << " /path/msalt_raw /path/sustech_format\n";
        std::cerr << "Example (Reverse): " << argv[0] << " /path/sustech_format /path/msalt_bench --reverse\n";
        return EXIT_FAILURE;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    LabelConverter converter(input_dir, output_dir, reverse_mode);

    if (!converter.processLabels()) {
        std::cerr << "[ERROR] Label conversion pipeline aborted.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
