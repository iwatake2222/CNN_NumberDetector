cmake_minimum_required(VERSION 2.8)
project(NumberDetector)

# Switch build target
set(BUILD_TARGET PC CACHE STRING "Build target?")

# Common build settings
include(cmakes/common.cmake)

# Build settings for each target
if(${BUILD_TARGET} STREQUAL PC)
	if(WIN32)
		message(FATAL_ERROR "[BUILD] PC Windows is not supported")
	else()
		message("[BUILD] PC Linux")
		include(cmakes/PC_LINUX.cmake)
	endif()
elseif(${BUILD_TARGET} STREQUAL JETSON_NATIVE)
	message("[BUILD] Jetson Native")
	include(cmakes/JETSON_NATIVE.cmake)
elseif(${BUILD_TARGET} STREQUAL RASPI_NATIVE)
	message("[BUILD] Raspberry Pi Native")
	include(cmakes/RASPI_NATIVE.cmake)
elseif(${BUILD_TARGET} STREQUAL RASPI_CROSS)
	message("[BUILD] Raspberry Pi Cross")
	include(cmakes/RASPI_CROSS.cmake)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()

# Create Main project
add_executable(NumberDetector
	main.cpp
	utils.cc
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(NumberDetector PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(NumberDetector ${OpenCV_LIBS})
endif()

# For Tensorflow Lite and Edge TPU
if((${BUILD_TARGET} STREQUAL RASPI_NATIVE) OR (${BUILD_TARGET} STREQUAL RASPI_CROSS))
	target_link_libraries(NumberDetector ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/libedgetpu/libedgetpu_arm32.so)
	target_link_libraries(NumberDetector /usr/lib/arm-linux-gnueabihf/libc++.a)
	target_link_libraries(NumberDetector /usr/lib/arm-linux-gnueabihf/libc++abi.a)
	target_link_libraries(NumberDetector unwind)
	target_link_libraries(NumberDetector ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt_static/rpi_armv7l/libtensorflow-lite.a)
elseif(${BUILD_TARGET} STREQUAL JETSON_NATIVE)
	target_link_libraries(NumberDetector ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/libedgetpu/libedgetpu_arm64.so)
	target_link_libraries(NumberDetector ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt_static/aarch64/libtensorflow-lite.a)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()

target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/libedgetpu/)
target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/eigen )
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/absl)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorlow/lite/tools/make/downloads/neon_2_sse)
# target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src)
target_include_directories(NumberDetector PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

