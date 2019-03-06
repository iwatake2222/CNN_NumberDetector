#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include "tf_utils.hpp"

#define MODEL_FILENAME RESOURCE_DIR"conv_mnist.pb"

static int displayGraphInfo()
{
	TF_Graph *graph = tf_utils::LoadGraphDef(MODEL_FILENAME);
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	size_t pos = 0;
	TF_Operation* oper;
	printf("--- graph info ---\n");
	while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
		printf("%s\n", TF_OperationName(oper));
	}
	printf("--- graph info ---\n");

	TF_DeleteGraph(graph);
	return 0;
}

int main()
{
	printf("Hello from TensorFlow C library version %s\n", TF_Version());

	/* read input image data */
	cv::Mat image = cv::imread(RESOURCE_DIR"4.jpg");
	cv::imshow("InputImage", image);
	
	/* convert to 28 x 28 grayscale image (normalized: 0 ~ 1.0) */
	cv::cvtColor(image, image, CV_BGR2GRAY);
	cv::resize(image, image, cv::Size(28, 28));
	image = ~image;
	cv::imshow("InputImage for CNN", image);
	image.convertTo(image, CV_32FC1, 1.0 / 255);

	/* get graph info */
	displayGraphInfo();

	TF_Graph *graph = tf_utils::LoadGraphDef(MODEL_FILENAME);
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	/* prepare input tensor */
	TF_Output input_op = { TF_GraphOperationByName(graph, "input_2_7"), 0 };
	if (input_op.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}

	const std::vector<std::int64_t> input_dims = { 1, 28, 28, 1 };
	std::vector<float> input_vals;
	image.reshape(0, 1).copyTo(input_vals);	// Mat to vector

	TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT,
		input_dims.data(), input_dims.size(),
		input_vals.data(), input_vals.size() * sizeof(float));

	/* prepare output tensor */
	TF_Output out_op = { TF_GraphOperationByName(graph, "dense_1_7/Softmax"), 0 };
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

	for (int i = 0; i < 10; i++) {
		printf("prob of %d: %.3f\n", i, probs[i]);
	}

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensor);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);

	cv::waitKey(0);
	return 0;
}