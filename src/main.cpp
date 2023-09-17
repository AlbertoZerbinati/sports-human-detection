#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./sports-human-detection <image_path> <model_path>"
                  << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = argv[2];

    std::cout << "Image Path: " << image_path << std::endl;
    std::cout << "Model Path: " << model_path << std::endl;

    // TODO

    return 0;
}