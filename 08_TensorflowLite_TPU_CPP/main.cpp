#include <stdio.h>
#include <string.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "edgetpu.h"
#include "utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

/*** Model parameters ***/
#define MODEL_FILENAME "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
// #define MODEL_FILENAME "mobilenet_v2_1.0_224_quant.tflite"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL 3


int main()
{
	/*** Create interpreter ***/
	/* read model */
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
	/* initialize edgetpu_context */
	edgetpu::EdgeTpuContext* edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext().release();
	/* create interpreter */
	std::unique_ptr<tflite::Interpreter> interpreter = coral::BuildEdgeTpuInterpreter(*model, edgetpu_context);

	/*** Read input image data ***/
	cv::Mat inputImage = cv::imread("parrot.jpg");
	cv::cvtColor(inputImage, inputImage, CV_BGR2RGB);
	cv::resize(inputImage, inputImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	std::vector<uint8_t> inputData(inputImage.data, inputImage.data + (inputImage.cols * inputImage.rows * inputImage.elemSize()));

	/*** Run inference ***/
	const auto& result = coral::RunInference(inputData, interpreter.get());

	/*** Retrieve result ***/
	// for (int i = 0; i < result.size(); i++) printf("%5d: %f\n", i, result[i]);
	auto it_a = std::max_element(result.begin(), result.end());
	printf("Max index: %ld (%.3f)\n", std::distance(result.begin(), it_a), *it_a);

	/*** Measure calculation time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < 100; i++) {
		coral::RunInference(inputData, interpreter.get());
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Calculation time = %f [sec]\n", timeSpan.count() / 100);
	
	return 0;
}

#if 0
how to build
mkdir build && cd build
wget https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
wget http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xfz mobilenet_v2_1.0_224_quant.tgz
wget https://coral.withgoogle.com/static/docs/images/parrot.jpg
cp ../external_libs/edgetpu-native/libedgetpu/libedgetpu_arm64.so libedgetpu.so.1
# cp ../external_libs/edgetpu-native/libedgetpu/libedgetpu_arm32.so libedgetpu.so
cmake .. -DBUILD_TARGET=JETSON_NATIVE
# cmake .. -DBUILD_TARGET=RASPI_NATIVE
make -j2
sudo LD_LIBRARY_PATH=./ ./NumberDetector
#endif
