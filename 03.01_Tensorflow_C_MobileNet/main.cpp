#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include "tf_utils.hpp"

// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
// http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
#define MODEL_FILENAME RESOURCE_DIR"mobilenet_v1_1.0_224_frozen.pb"

int main()
{
	printf("Hello from TensorFlow C library version %s\n", TF_Version());

	/* read input image data */
	cv::Mat image = cv::imread(RESOURCE_DIR"parrot.jpg");
	// cv::imshow("InputImage", image);
	
	/* Fit to model input format */
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	cv::resize(image, image, cv::Size(224, 224));
	// cv::imshow("InputImage for CNN", image);
	image.convertTo(image, CV_32FC1, 1.0 / 255);

	TF_Graph *graph = tf_utils::LoadGraphDef(MODEL_FILENAME);
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}
	
	/* prepare input tensor */
	TF_Output input_op = { TF_GraphOperationByName(graph, "input"), 0 };
	if (input_op.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}
	
	const std::vector<std::int64_t> input_dims = { 1, 224, 224, 3 };
	std::vector<float> input_vals;
	image.reshape(1, 1).copyTo(input_vals);	// Mat to vector

	TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT,
		input_dims.data(), input_dims.size(),
		input_vals.data(), input_vals.size() * sizeof(float));

	/* prepare output tensor */
	TF_Output out_op = { TF_GraphOperationByName(graph, "MobilenetV1/Predictions/Reshape_1"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't init out_op" << std::endl;
		return 3;
	}

	TF_Tensor* output_tensor = nullptr;

	/* prepare session */
	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteStatus(status);
		return 4;
	}

	/* run session */
	TF_SessionRun(sess,
		nullptr, // Run options.
		&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
		&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		status // Output status.
	);

	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error run session";
		TF_DeleteStatus(status);
		return 5;
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error close session";
		TF_DeleteStatus(status);
		return 6;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error delete session";
		TF_DeleteStatus(status);
		return 7;
	}

	const auto probs = static_cast<float*>(TF_TensorData(output_tensor));

	float max = 0;
	int maxIndex = -1;
	for (int i = 0; i < 1000; i++) {
		if (probs[i] > max) {
			max = probs[i];
			maxIndex = i;
		}
		// printf("prob of %d: %f\n", i, probs[i]);
	}
	printf("%d:  %.3f\n", maxIndex, max);

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensor);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);

	cv::waitKey(0);
	return 0;
}
