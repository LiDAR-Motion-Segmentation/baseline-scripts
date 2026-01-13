#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct AppConfig{
    fs::path input_dir;
    fs::path output_file = "output_500_frames.mp4";
    int fps=30;
    // Optional frame limit 
    std::optional<size_t> frame_limit = std::nullopt;
    
    static void print_usage(const char* prog_name) {
        std::cerr << "Usage: " << prog_name << " <input_dir> [options]\n"
                  << "Options:\n"
                  << "  -o <path>       Output video path (default: output.mp4)\n"
                  << "  -fps <int>      Frames per second (default: 30)\n"
                  << "  -n <int>        Limit number of frames to process\n"
                  << "  -h, --help      Show this help message\n";
    }
};

class ImageSequenceReader {
    public:
        explicit ImageSequenceReader(const fs::path& dir_path) : directory(dir_path){
            if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
                throw std::runtime_error("Invalid input directory: " + dir_path.string());
            }
        }

        void scan_directory() {
            const std::vector<std::string> valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
            image_file.clear();

            std::cout << "[INFO] Scanning directory: " << directory << std::endl;

            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file()){
                    std::string ext = entry.path().extension().string();
                    // lowercase conversion for comparison
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    for (const auto& valid : valid_exts) {
                        if (ext == valid){
                            image_file.push_back(entry.path());
                            break;
                        }
                    }

                }
            }

            // sort files to ensure ordering
            std::sort(image_file.begin(), image_file.end());

            if (image_file.empty()){
                throw std::runtime_error("no images found in the directory");
            }

            std::cout << "[INFO] Found " << image_file.size() << " images" << std::endl;
        }

        // applies modular frame limit logic
        void apply_limit(std::optional<size_t> limit){
            if (limit.has_value() && limit.value() < image_file.size()){
                std::cout << "[INFO] imiting the number of frames" << limit.value() << " frames." << std::endl;
                image_file.resize(limit.value());
            }
        }

        const std::vector<fs::path>& get_files() const {
            return image_file;
        }
        
    private:
        fs::path directory;
        std::vector<fs::path> image_file;    
};

class VideoEncoder {
    public:
        VideoEncoder(const fs::path& out_path, int fps, cv::Size resolution)
        : output_path(out_path), fps(fps), frame_size(resolution) {
            // use mp4 for broad comaptibility
            int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
            writer.open(output_path.string(), fourcc, fps, frame_size, true);

            if (!writer.isOpened()) {
                throw std::runtime_error("Failed to open VideoWriter for file: " + output_path.string());
            }
        }

        void write_frame(const cv::Mat& frame){
            if (frame.empty()) return;

            // ensure frame matches the video resolution 
            if (frame.size() != frame_size){
                cv::Mat resized;
                cv::resize(frame, resized, frame_size);
                writer.write(resized);
            }
            else {
                writer.write(frame);
            }
        }

        ~VideoEncoder() {
            if (writer.isOpened()){
                writer.release();
                std::cout << "[INFO] Video saved successfully: " << output_path << std::endl;
            }
        }

    private:
        fs::path output_path;
        int fps;
        cv::Size frame_size;
        cv::VideoWriter writer;
};

int main(int argc, char** argv){
    if (argc < 2) {
        AppConfig::print_usage(argv[0]);
        return 1;
    }
    AppConfig config;
    config.input_dir = argv[1];

    for (int i = 2; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc){
            config.output_file = argv[++i];
        } 
        else if (arg == "-fps" && i + 1 < argc){
            config.fps = std::stoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc){
            config.frame_limit = std::stoul(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            AppConfig::print_usage(argv[0]);
            return 0;
        }
    }

    try{
        ImageSequenceReader reader(config.input_dir);
        reader.scan_directory();
        reader.apply_limit(config.frame_limit);
        const auto& files = reader.get_files();

        // checking video encoder for the first time
        cv::Mat first_frame = cv::imread(files[0].string());
        if (first_frame.empty()) {
            throw std::runtime_error("Could not read first image to determine resolution.");
        }

        VideoEncoder encoder(config.output_file, config.fps, first_frame.size());

        // processing loop
        std::cout << "[INFO] Encoding started" << std::endl;
        size_t count = 0;
        size_t total = files.size();

        for (const auto& file_path : files) {
            cv::Mat frame = cv::imread(file_path.string());
            if (frame.empty()) {
                std::cerr << "[WARN] Skipping corrupt/unreadable frame: " << file_path << std::endl;
                continue;
            }

            encoder.write_frame(frame);

            count++;
            if (count % 10 == 0 || count == total) {
                float progress = (float)count / total * 100.0f;
                std::cout << "\r[INFO] Progress: " << count << "/" << total << " (" << (int)progress << "%)" << std::flush;
            }
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e){
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}