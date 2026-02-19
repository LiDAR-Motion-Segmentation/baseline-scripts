/**
 * @file nuscenes_to_sustech.cpp
 * @brief Converts nuScenes mini-split sequences into SUSTechPOINTS format.
 */
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <iomanip>
#include <map>
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

class NuScenesToSustechConverter {
private:
    std::string dataset_path_;
    std::string output_path_;
    std::string sequence_name_;

    json readJson(const std::string& filename) const {
        fs::path file_path = fs::path(dataset_path_) / "v1.0-mini" / filename;
        std::ifstream file(file_path);
        if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON: " + file_path.string());
        }
    json j;
    file >> j;
    return j;
    }
    // Helper to convert nuScenes 5-channel .bin to 4-channel .pcd
    void convertBinToPcd(const fs::path& input_bin, const fs::path& output_pcd) const {
        std::ifstream in(input_bin, std::ios::binary);
        if (!in) {
            std::cerr << "[WARNING] Missing LiDAR bin: " << input_bin << "\n";
            return;
        }
        in.seekg(0, std::ios::end);
        size_t size = in.tellg();
        in.seekg(0, std::ios::beg);

        std::vector<float> points(size / sizeof(float));
        in.read(reinterpret_cast<char*>(points.data()), size);

        // nuScenes format: x, y, z, intensity, ring_index (5 floats)
        size_t num_points = points.size() / 5;

        std::ofstream out(output_pcd, std::ios::binary);

        // Standard PCD Header
        out << "# .PCD v0.7 - Point Cloud Data file format\n"
            << "VERSION 0.7\n"
            << "FIELDS x y z intensity\n"
            << "SIZE 4 4 4 4\n"
            << "TYPE F F F F\n"
            << "COUNT 1 1 1 1\n"
            << "WIDTH " << num_points << "\n"
            << "HEIGHT 1\n"
            << "VIEWPOINT 0 0 0 1 0 0 0\n"
            << "POINTS " << num_points << "\n"
            << "DATA binary\n";

        // Write binary data (ignoring the 5th float)
        for (size_t i = 0; i < num_points; ++i) {
            out.write(reinterpret_cast<const char*>(&points[i * 5]), 4 * sizeof(float));
        }
    }
    
public:
    /**
     * @brief Constructor for the converter pipeline.
     * @param dataset_path Root path to the nuScenes mini dataset.
     * @param output_path Where the SUSTechPOINTS formatted data will be saved.
     * @param sequence_name The specific scene to extract (e.g., "scene-0061").
     */
        NuScenesToSustechConverter(
            const std::string& dataset_path,
            const std::string& output_path,
            const std::string& sequence_name):
          dataset_path_(dataset_path), 
          output_path_(output_path), 
          sequence_name_(sequence_name) {}

    /**
     * @brief Creates the target directory tree required by SUSTechPOINTS.
     * @return true if successful, false otherwise.
     */
    bool createDirectoryStructure() const {
        // Define the target subdirectories based on the required tree
        const std::vector<std::string> subdirs = {
            "calib/camera",
            "camera/CAM_BACK",
            "camera/CAM_BACK_LEFT",
            "camera/CAM_BACK_RIGHT",
            "camera/CAM_FRONT",
            "camera/CAM_FRONT_LEFT",
            "camera/CAM_FRONT_RIGHT",
            "label",
            "lidar"
        };

        try {
            // Create the main output directory for this specific sequence
            fs::path seq_dir = fs::path(output_path_) / sequence_name_;
            
            if (!fs::exists(seq_dir)) {
                fs::create_directories(seq_dir);
            }

            // Generate all necessary nested subdirectories
            for (const auto& subdir : subdirs) {
                fs::path full_path = seq_dir / subdir;
                fs::create_directories(full_path);
            }
            
            std::cout << "[INFO] Successfully created directory structure at: " << seq_dir << "\n";
            return true;

        } catch (const fs::filesystem_error& e) {
            std::cerr << "[ERROR] Filesystem error while creating directories: " << e.what() << "\n";
            return false;
        }
    }

    bool extractSensorData() const {
        try {
            std::cout << "[INFO] Loading nuScenes JSON metadata (this may take a moment)...\n";
            json scenes = readJson("scene.json");
            json samples = readJson("sample.json");
            json sample_data = readJson("sample_data.json");

            std::string first_sample_token = "";
            for (const auto& scene : scenes) {
                if (scene["name"] == sequence_name_) {
                    first_sample_token = scene["first_sample_token"];
                    break;
                }
            }

            if (first_sample_token.empty()) {
                std::cerr << "[ERROR] Sequence " << sequence_name_ << " not found in scene.json\n";
                return false;
            }

            std::map<std::string, json> sample_data_map;
            for (const auto& sd : sample_data) {
                sample_data_map[sd["token"]] = sd;
            }

            std::map<std::string, json> sample_map;
            for (const auto& s : samples) {
                sample_map[s["token"]] = s;
            }

            std::string current_sample_token = first_sample_token;
            int frame_index = 0;

            std::vector<std::string> sensors = {
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "LIDAR_TOP"
            };

            while (!current_sample_token.empty()) {
                const auto& sample = sample_map[current_sample_token];

                std::ostringstream frame_str;
                frame_str << std::setw(6) << std::setfill('0') << frame_index;
                std::string frame_name = frame_str.str();
                std::cout << "[INFO] Processing frame: " << frame_name << "\n";

                for (const auto& sensor : sensors) {
                    if (sample["data"].contains(sensor)) {
                        std::string data_token = sample["data"][sensor];
                        std::string filename = sample_data_map[data_token]["filename"];

                        fs::path source_path = fs::path(dataset_path_) / filename;
                        if (sensor == "LIDAR_TOP") {
                            fs::path target_path = fs::path(output_path_) / sequence_name_ / "lidar" / (frame_name + ".pcd");
                            convertBinToPcd(source_path, target_path);
                        } 
                        else {
                            fs::path target_path = fs::path(output_path_) / sequence_name_ / "camera" / sensor / (frame_name + ".png");
                            if (fs::exists(source_path)) {
                                fs::copy_file(source_path, target_path, fs::copy_options::overwrite_existing);
                        }
                    }
                }
            }

            current_sample_token = sample["next"];
            frame_index++;
        }

        std::cout << "[INFO] Successfully extracted data for " << frame_index << " frames.\n";
            return true;
        }

        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during data extraction: " << e.what() << "\n";
            return false;
        }
    }
};

int main(int argc, char* argv[]) {
    // Basic Argument Parsing
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <nuscenes_dataset_path> <output_path> <sequence_name>\n";
        std::cerr << "Example: " << argv[0] << " /data/sets/nuscenes /workspace/sustech_data scene-0061\n";
        return EXIT_FAILURE;
    }

    std::string dataset_path = argv[1];
    std::string output_path  = argv[2];
    std::string sequence_name = argv[3];

    // Validate Input Dataset Path
    if (!fs::exists(dataset_path)) {
        std::cerr << "[ERROR] Provided nuScenes dataset path does not exist: " << dataset_path << "\n";
        return EXIT_FAILURE;
    }

    // Initialize the Pipeline
    NuScenesToSustechConverter converter(dataset_path, output_path, sequence_name);
    
    if (!converter.createDirectoryStructure()) {
        std::cerr << "[ERROR] Pipeline aborted during directory setup.\n";
        return EXIT_FAILURE;
    }

    if (!converter.extractSensorData()) {
        std::cerr << "[ERROR] Pipeline aborted during sensor data extraction.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] Step 1 complete. Ready to populate files.\n";
    return EXIT_SUCCESS;
};