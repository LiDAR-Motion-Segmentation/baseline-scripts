/**
 * @file nuscenes_to_sustech.cpp
 * @brief Converts nuScenes mini-split sequences into SUSTechPOINTS format.
 * running instruction
 * g++ -std=c++17 -O3 -I. nuscenes_to_sustech.cpp -o nusc_converter
 * ./nusc_converter /home/soumoroy/Downloads/v1.0-mini/ /home/soumoroy/Downloads/annotations/aug5_sustech/nuscenes_v2/ scene-0103
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
using ordered_json = nlohmann::ordered_json;

struct Vec3 { double x, y, z; };
struct Quat { double w, x, y, z; };

Vec3 rotate_vector(const Quat& q, const Vec3& v) {
    double tx = 2.0 * (q.y * v.z - q.z * v.y);
    double ty = 2.0 * (q.z * v.x - q.x * v.z);
    double tz = 2.0 * (q.x * v.y - q.y * v.x);
    return {
        v.x + q.w * tx + (q.y * tz - q.z * ty),
        v.y + q.w * ty + (q.z * tx - q.x * tz),
        v.z + q.w * tz + (q.x * ty - q.y * tx)
    };
}

Quat inverse(const Quat& q) {
    return {q.w, -q.x, -q.y, -q.z};
}

Quat multiply(const Quat& q1, const Quat& q2) {
    return {
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    };
}

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

    std::string mapCategory(const std::string& cat_name) const {
        if (cat_name.find("human") != std::string::npos) return "moving_people";
        if (cat_name.find("vehicle.car") != std::string::npos) return "moving_car";
        if (cat_name.find("vehicle.truck") != std::string::npos) return "moving_truck";
        if (cat_name.find("vehicle.bus") != std::string::npos) return "moving_bus";
        if (cat_name.find("vehicle.bicycle") != std::string::npos || 
            cat_name.find("vehicle.motorcycle") != std::string::npos) return "moving_cyclist";
        if (cat_name.find("vehicle.construction") != std::string::npos) return "moving_construction_vehicle";
        return "unknown";
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

    bool generateFilenameLists() const {
        std::cout << "[INFO] Generating filename list text files...\n";
        try {
            fs::path seq_dir = fs::path(output_path_) / sequence_name_;
            fs::path lidar_dir = seq_dir / "lidar";

            int num_frames = 0;
            if (fs::exists(lidar_dir)) {
                for (const auto& entry : fs::directory_iterator(lidar_dir)) {
                    if (entry.path().extension() == ".pcd") {
                        num_frames++;
                    }
                }
            }
            else {
                std::cerr << "[ERROR] Lidar directory not found. Run extractSensorData first.\n";
                return false;
            }

            if (num_frames == 0) {
                std::cerr << "[WARNING] No frames found to generate lists for.\n";
                return false;
            }

            fs::path pc_file_path = seq_dir / "point_cloud_filenames.txt";
            std::ofstream pc_file(pc_file_path);
            for (int i = 0; i < num_frames; ++i) {
                std::ostringstream frame_str;
                frame_str << std::setw(6) << std::setfill('0') << i;
                pc_file << "lidar/" << frame_str.str() << ".pcd\n";
            }
            std::cout << "[INFO] Wrote point_cloud_filenames.txt (" << num_frames << " frames)\n";

            std::vector<std::string> cameras = {
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
            };

            for (const auto& cam : cameras) {
                fs::path cam_file_path = seq_dir / (cam + "_filenames.txt");
                std::ofstream cam_file(cam_file_path);
                for (int i = 0; i < num_frames; ++i) {
                    std::ostringstream frame_str;
                    frame_str << std::setw(6) << std::setfill('0') << i;
                    cam_file << "camera/" << cam << "/" << frame_str.str() << ".png\n";
                }
            }
            std::cout << "[INFO] Wrote all 6 camera filename lists.\n";

            fs::path ann_file_path = seq_dir / "annotation_filenames.txt";
            std::ofstream ann_file(ann_file_path);
            for (int i = 0; i < num_frames; ++i) {
                std::ostringstream frame_str;
                frame_str << std::setw(6) << std::setfill('0') << i;
                ann_file << "label/" << frame_str.str() << ".json\n";
            }
            std::cout << "[INFO] Wrote annotation_filenames.txt\n";

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during filename list generation: " << e.what() << "\n";
            return false;
        }
    }

    bool extractGTAnnotations() const {
        std::cout << "[INFO] Extracting Raw nuScenes GT Annotations...\n";
        try {
            json scenes = readJson("scene.json");
            json samples = readJson("sample.json");
            json sample_data = readJson("sample_data.json");
            json annotations = readJson("sample_annotation.json");
            json ego_poses = readJson("ego_pose.json");
            json calib_sensors = readJson("calibrated_sensor.json");
            json categories = readJson("category.json");
            json instances = readJson("instance.json"); // Added instance mapping!

            std::map<std::string, json> category_map, ego_map, calib_map, sample_map, instance_map;
            for (const auto& c : categories) category_map[c["token"].get<std::string>()] = c;
            for (const auto& e : ego_poses) ego_map[e["token"].get<std::string>()] = e;
            for (const auto& cs : calib_sensors) calib_map[cs["token"].get<std::string>()] = cs;
            for (const auto& s : samples) sample_map[s["token"].get<std::string>()] = s;
            for (const auto& i : instances) instance_map[i["token"].get<std::string>()] = i;

            std::map<std::string, std::vector<json>> sample_to_data;
            for (const auto& sd : sample_data) {
                if (sd.contains("is_key_frame") && sd["is_key_frame"] == true && sd.contains("sample_token")) {
                    sample_to_data[sd["sample_token"].get<std::string>()].push_back(sd);
                }
            }

            std::map<std::string, std::vector<json>> sample_to_anns;
            for (const auto& a : annotations) {
                if (a.contains("sample_token")) {
                    sample_to_anns[a["sample_token"].get<std::string>()].push_back(a);
                }
            }

            std::string first_sample_token = "";
            for (const auto& scene : scenes) {
                if (scene.contains("name") && scene["name"] == sequence_name_) {
                    first_sample_token = scene["first_sample_token"];
                    break;
                }
            }
            if (first_sample_token.empty()) return false;

            std::string current_sample_token = first_sample_token;
            int frame_index = 0;

            while (!current_sample_token.empty()) {
                if (sample_map.find(current_sample_token) == sample_map.end()) break;
                const auto& sample = sample_map[current_sample_token];
                
                json lidar_data;
                auto data_it = sample_to_data.find(current_sample_token);
                if (data_it != sample_to_data.end()) {
                    for (const auto& sd : data_it->second) {
                        std::string filename = sd["filename"].get<std::string>();
                        if (filename.find("LIDAR_TOP") != std::string::npos) {
                            lidar_data = sd;
                            break;
                        }
                    }
                }

                if (lidar_data.empty()) {
                    current_sample_token = (sample.contains("next") && !sample["next"].is_null()) ? sample["next"].get<std::string>() : "";
                    frame_index++;
                    continue;
                }

                json ego = ego_map[lidar_data["ego_pose_token"].get<std::string>()];
                json cs = calib_map[lidar_data["calibrated_sensor_token"].get<std::string>()];

                Vec3 T_ego = {ego["translation"][0], ego["translation"][1], ego["translation"][2]};
                Quat Q_ego = {ego["rotation"][0], ego["rotation"][1], ego["rotation"][2], ego["rotation"][3]};

                Vec3 T_sens = {cs["translation"][0], cs["translation"][1], cs["translation"][2]};
                Quat Q_sens = {cs["rotation"][0], cs["rotation"][1], cs["rotation"][2], cs["rotation"][3]};

                nlohmann::ordered_json output_json = json::array();
                int obj_id_counter = 1;

                auto anns_it = sample_to_anns.find(current_sample_token);
                if (anns_it != sample_to_anns.end()) {
                    for (const auto& ann : anns_it->second) {
                        Vec3 box_center = {ann["translation"][0], ann["translation"][1], ann["translation"][2]};
                        Quat box_rot = {ann["rotation"][0], ann["rotation"][1], ann["rotation"][2], ann["rotation"][3]};

                        // Global -> Ego
                        Vec3 p_ego = rotate_vector(inverse(Q_ego), {box_center.x - T_ego.x, box_center.y - T_ego.y, box_center.z - T_ego.z});
                        Quat rot_ego = multiply(inverse(Q_ego), box_rot);

                        // Ego -> Sensor
                        Vec3 p_sens = rotate_vector(inverse(Q_sens), {p_ego.x - T_sens.x, p_ego.y - T_sens.y, p_ego.z - T_sens.z});
                        Quat rot_sens = multiply(inverse(Q_sens), rot_ego);

                        double qw = rot_sens.w, qx = rot_sens.x, qy = rot_sens.y, qz = rot_sens.z;
                        double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

                        nlohmann::ordered_json sustech_box;
                        sustech_box["obj_id"] = std::to_string(obj_id_counter++);
                        
                        // NEW INSTANCE/CATEGORY LOOKUP
                        std::string inst_token = ann["instance_token"].get<std::string>();
                        std::string cat_token = instance_map[inst_token]["category_token"].get<std::string>();
                        std::string cat_name = category_map[cat_token]["name"].get<std::string>();
                        
                        sustech_box["obj_type"] = mapCategory(cat_name);
                        
                        // RAW nuScenes Sensor Coordinates (No Swaps)
                        sustech_box["psr"]["position"]["x"] = p_sens.x;
                        sustech_box["psr"]["position"]["y"] = p_sens.y;
                        sustech_box["psr"]["position"]["z"] = p_sens.z;

                        // nuScenes scale: [width, length, height]
                        sustech_box["psr"]["scale"]["x"] = ann["size"][0]; 
                        sustech_box["psr"]["scale"]["y"] = ann["size"][1]; 
                        sustech_box["psr"]["scale"]["z"] = ann["size"][2]; 

                        sustech_box["psr"]["rotation"]["x"] = 0.0;
                        sustech_box["psr"]["rotation"]["y"] = 0.0;
                        sustech_box["psr"]["rotation"]["z"] = yaw;

                        output_json.push_back(sustech_box);
                    }
                }

                std::ostringstream frame_str;
                frame_str << std::setw(6) << std::setfill('0') << frame_index;
                fs::path out_file = fs::path(output_path_) / sequence_name_ / "label" / (frame_str.str() + ".json");
                
                std::ofstream o(out_file);
                o << std::setw(2) << output_json << std::endl;

                current_sample_token = (sample.contains("next") && !sample["next"].is_null()) ? sample["next"].get<std::string>() : "";
                frame_index++;
            }
            std::cout << "[INFO] Finished extracting Raw Ground Truth labels.\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during GT extraction: " << e.what() << "\n";
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

    if (!converter.generateFilenameLists()){
        std::cerr << "[ERROR] Pipeline aborted during file list generation.\n";
        return EXIT_FAILURE;
    }

    if (!converter.extractGTAnnotations()){
        std::cerr << "[ERROR] Pipeline aborted during GT labels generation.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] Execution finished successfully!\n";
    return EXIT_SUCCESS;
};