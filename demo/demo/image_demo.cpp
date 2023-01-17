#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "Yolo3Detection.h"
// todo: add yaml include


int main(int argc, char* argv[]) {
	// Readin config file .rt filename from arguments
	if (argc < 1) {
		std::cerr << "A config file must be provided.\n";
		exit(1);
	}
	std::string const config_file = std::string(argv[1]);

	YAML::Node config = YAMLloadConf(config_file);
	if (!config) {
		std::cerr << "Unable to load config file: " << config_file << "\n";
		exit(2);
	}

	std::string const network_rt_file = YAMLgetConf<std::string>(config, "network_rt_file", "yolo4tiny_fp32.rt");
	if (!fileExist(network_rt_file.c_str())) {
		std::cerr << "The given network does not exist, create the rt file first: " << network_rt_file << "\n";
		exit(3);
	}
	std::string const input_filename = YAMLgetConf<std::string>(config, "input_filename", "");
	if (!fileExist(input_filename.c_str())) {
		std::cerr << "The given input video does not exist: " << input_filename << "\n";
		exit(4);
	}
		
	std::string const output_filename = YAMLgetConf<std::string>(config, "output_filename", "");
	char const ntype = YAMLgetConf<char>(config, "ntype", 'y');
	int const class_count = YAMLgetConf<int>(config, "class_count", 80);
	int const batch_size = YAMLgetConf<int>(config, "n_batch", 1);
	float const confidence_threshold = YAMLgetConf<float>(config, "n_batch", 0.3);

	// Create a YOLO network and load the .rt file
	tk::dnn::Yolo3Detection network;
	network.init(network_rt_file, class_count, batch_size, confidence_threshold);

	// Load the test image
	cv::Mat input_image = cv::imread(input_filename, cv::IMREAD_COLOR);

	// Perform inference on the image
	std::vector<cv::Mat> input_images{ input_image };
	network.update(input_images, batch_size);

	// Print the detection results
	unsigned int batch_index = 0;
	for (auto const& batch : network.batchDetected) {
		std::cout << "--- Batch " << batch_index << "---\n";
		unsigned int detection_index = 0;
		for (auto const& detection : batch) {
			std::cout << "Detection " << detection_index << ":\n";
			std::cout << "- Class ID: " << detection.cl << "\n";
			std::cout << "- Bounding Box: " << detection.x << " " << detection.y << " " << detection.w << " " << detection.h << "\n";
			std::cout << "- Probability: " << detection.prob << "\n";

			std::cout << "- Distribution (" << detection.probs.size() << "): ";
			for (auto const& class_probability : detection.probs) {
				std::cout << class_probability << " ";
			}
			std::cout << "\n";

			detection_index++;
		}
		batch_index++;
	}

	// Draw the results on a new image
	cv::Mat output_image = input_image.clone();
	std::vector<cv::Mat> output_images{ output_image };
	network.draw(output_images);

	// Save the final image
	cv::imwrite(output_filename, output_image);

	return 0;
}
