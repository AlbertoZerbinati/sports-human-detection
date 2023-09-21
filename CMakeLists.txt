# Alberto Zerbinati

cmake_minimum_required(VERSION 2.8)
project(sports_human_detection)

# Specify the path to LibTorch (replace with your actual global path)
list(APPEND CMAKE_PREFIX_PATH "/workspace/libtorch")

# Find packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*/*.cpp")

# file(GLOB_RECURSE TEST_SOURCES_1 "test/PeopleDetectorTest.cpp" "src/*/*.cpp")
# file(GLOB_RECURSE TEST_SOURCES_2 "test/PeopleDetectorAndSegmentationTest.cpp" "src/*/*.cpp")
# file(GLOB_RECURSE TEST_SOURCES_3 "test/DetectorSegmentationTeamTest.cpp" "src/*/*.cpp")
# file(GLOB_RECURSE TEST_SOURCES_4 "test/DetectSegmentTeamFieldTest.cpp" "src/*/*.cpp")

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add tests executable
# add_executable(test_detector ${TEST_SOURCES_1})
# add_executable(test_detector_segmentation ${TEST_SOURCES_2})
# add_executable(test_detector_segmentation_team ${TEST_SOURCES_3})
# add_executable(test_detect_segment_team_field ${TEST_SOURCES_4})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# target_link_libraries(test_detector ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_link_libraries(test_detector_segmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_link_libraries(test_detector_segmentation_team ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_link_libraries(test_detect_segment_team_field ${OpenCV_LIBS} ${TORCH_LIBRARIES})
