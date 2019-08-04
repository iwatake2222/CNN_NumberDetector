cmake_minimum_required(VERSION 2.8)
project(TensorFlowC)

# Create Main project
add_executable(TensorFlowC
	main.cpp
	tf_utils.cpp
	tf_utils.hpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(TensorFlowC PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(TensorFlowC ${OpenCV_LIBS})
endif()

# For Tensorflow
target_link_libraries(TensorFlowC ${CMAKE_SOURCE_DIR}/libtensorflow/lib/libtensorflow.so)
target_link_libraries(TensorFlowC ${CMAKE_SOURCE_DIR}/libtensorflow/lib/libtensorflow_framework.so)
target_include_directories(TensorFlowC PUBLIC libtensorflow/include)
# file(COPY libtensorflow/lib/libtensorflow.so DESTINATION ${PROJECT_BINARY_DIR})
# file(COPY libtensorflow/lib/libtensorflow_framework.so DESTINATION ${PROJECT_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++")


# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")

