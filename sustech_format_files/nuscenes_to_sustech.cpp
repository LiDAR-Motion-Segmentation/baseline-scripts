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
                if (scene.contains("name") && scene["name"] == sequence_name_) {
                    first_sample_token = scene["first_sample_token"];
                    break;
                }
            }

            if (first_sample_token.empty()) {
                std::cerr << "[ERROR] Sequence " << sequence_name_ << " not found in scene.json\n";
                return false;
            }

            std::map<std::string, std::vector<json>> sample_to_data;
            for (const auto& sd : sample_data) {
                if (sd.contains("is_key_frame") && sd["is_key_frame"] == true) {
                    if (sd.contains("sample_token")) {
                        std::string s_token = sd["sample_token"];
                        sample_to_data[s_token].push_back(sd);
                    }
                }
            }

            std::map<std::string, json> sample_map;
            for (const auto& s : samples) {
                if (s.contains("token")) {
                    sample_map[s["token"]] = s;
                }
            }

            std::string current_sample_token = first_sample_token;
            int frame_index = 0;

            std::vector<std::string> sensors = {
                "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_FRONT",
                "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK", "LIDAR_TOP"
            };

            while (!current_sample_token.empty()) {
                if (sample_map.find(current_sample_token) == sample_map.end()) {
                    break; // Safety break
                }

                const auto& sample = sample_map[current_sample_token];

                std::ostringstream frame_str;
                frame_str << std::setw(6) << std::setfill('0') << frame_index;
                std::string frame_name = frame_str.str();
                std::cout << "[INFO] Processing frame: " << frame_name << "\n";

                auto data_it = sample_to_data.find(current_sample_token);
                if (data_it != sample_to_data.end()) {
                    for (const auto& sd : data_it->second) {
                        std::string filename = sd["filename"];
                        
                        std::string current_sensor = "";
                        for (const auto& sensor : sensors) {
                            if (filename.find(sensor) != std::string::npos) {
                                current_sensor = sensor;
                                break;
                            }
                        }

                        if (!current_sensor.empty()) {
                            fs::path source_path = fs::path(dataset_path_) / filename;
                            
                            if (current_sensor == "LIDAR_TOP") {
                                fs::path target_path = fs::path(output_path_) / sequence_name_ / "lidar" / (frame_name + ".pcd");
                                convertBinToPcd(source_path, target_path);
                            } else {
                                fs::path target_path = fs::path(output_path_) / sequence_name_ / "camera" / current_sensor / (frame_name + ".png");
                                if (fs::exists(source_path)) {
                                    fs::copy_file(source_path, target_path, fs::copy_options::overwrite_existing);
                                }
                            }
                        }
                    }
                }

                // Safely grab the next token in the sequence list
                if (sample.contains("next") && !sample["next"].is_null() && !sample["next"].get<std::string>().empty()) {
                    current_sample_token = sample["next"];
                } else {
                    current_sample_token = ""; // End loop
                }
                
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

    bool extractCalibData() const {
        std::cout << "[INFO] Extracting Camera Calibration Data...\n";
        try {
            json scenes = readJson("scene.json");
            json sample_data = readJson("sample_data.json");
            json calib_sensors = readJson("calibrated_sensor.json");

            std::string first_sample_token = "";
            for (const auto& scene : scenes) {
                if (scene.contains("name") && scene["name"] == sequence_name_) {
                    first_sample_token = scene["first_sample_token"];
                    break;
                }
            }

            std::map<std::string, json> calib_map;
            for (const auto& cs : calib_sensors) {
                if (cs.contains("token")) calib_map[cs["token"]] = cs;
            }

            std::vector<std::string> cameras = {
                "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_FRONT",
                "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"
            };

            // Keep track of which cameras we've written calibration for
            std::map<std::string, bool> calib_written;

            for (const auto& sd : sample_data) {
                if (sd.contains("sample_token") && sd["sample_token"] == first_sample_token) {
                    std::string filename = sd["filename"];
                    std::string current_cam = "";
                    for (const auto& cam : cameras) {
                        if (filename.find(cam) != std::string::npos) {
                            current_cam = cam;
                            break;
                        }
                    }

                    // If it's a camera and we haven't written its JSON yet
                    if (!current_cam.empty() && !calib_written[current_cam]) {
                        std::string calib_token = sd["calibrated_sensor_token"];
                        json calib_info = calib_map[calib_token];

                        // extract data
                        int width = sd["width"];
                        int height = sd["height"];
                        auto K = calib_info["camera_intrinsic"];

                        // Format output to exactly match SUSTechPOINTS expectation
                        nlohmann::ordered_json output_json;
                        output_json["width"] = width;
                        output_json["height"] = height;
                        output_json["fx"] = K[0][0];
                        output_json["fy"] = K[1][1];
                        output_json["cx"] = K[0][2];
                        output_json["cy"] = K[1][2];
                        output_json["skew"] = K[0][1];
                        output_json["distortion"] = {-0.054603107273578644, 
                            0.06334752589464188, 
                            0.00022518340847454965, 
                            0.0002921034465543926, 
                            -0.020296046510338783}; // Example standard distortion to ensure 5 floats

                        // Write to calib/camera/CAM_NAME.json
                        fs::path out_file = fs::path(output_path_) / sequence_name_ / "calib" / "camera" / (current_cam + ".json");
                        std::ofstream o(out_file);
                        o << std::setw(2) << output_json << std::endl; // Indent with 2 spaces

                        calib_written[current_cam] = true;
                        std::cout << "[INFO] Wrote calibration for " << current_cam << "\n";
                    }
                }
            }
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during calibration extraction: " << e.what() << "\n";
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

    if (!converter.extractCalibData()){
        std::cerr << "[ERROR] Pipeline aborted during sensor calibration data extraction.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] Completed. Ready to populate files.\n";
    return EXIT_SUCCESS;
};