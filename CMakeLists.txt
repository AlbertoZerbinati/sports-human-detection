cmake_minimum_required(VERSION 3.10)
project(sports-human-detection)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
