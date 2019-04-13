#include <stdio.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define MODEL_FILENAME "conv_mnist.tflite"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


int main()
{

	/* tflite */
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
	TFLITE_MINIMAL_CHECK(model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	printf("=== Pre-invoke Interpreter State ===\n");
	tflite::PrintInterpreterState(interpreter.get());

	// Set data to input tensor
	float* input = interpreter->typed_input_tensor<float>(0);
	memset(input, 0, sizeof(float) * 1 * 28 * 28 * 1);

	// Run inference
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	printf("\n\n=== Post-invoke Interpreter State ===\n");
	tflite::PrintInterpreterState(interpreter.get());

	// Get data from output tensor
	float* probs = interpreter->typed_output_tensor<float>(0);
	for (int i = 0; i < 10; i++) {
		printf("prob of %d: %.3f\n", i, probs[i]);
	}

	return 0;
}
