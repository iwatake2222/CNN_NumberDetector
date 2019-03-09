cmake_minimum_required(VERSION 2.8)
project(NumberDetector)

# Create Main project
add_executable(NumberDetector
	main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(NumberDetector PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(NumberDetector ${OpenCV_LIBS})
endif()

# For Tensorflow Lite
target_link_libraries(NumberDetector ${PROJECT_SOURCE_DIR}/external_libs/tensorflow_prebuilt/linux-cpu-x86_64/libtensorflowlite.so)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/eigen )
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/absl)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorlow/lite/tools/make/downloads/neon_2_sse)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -lstdc++")

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")
