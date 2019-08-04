# 03.01_Tensorflow_C_MobileNet
Sample code to use TensorFlow C lib for Mobilenet

- Generate TensorFlow C library or get library from somewhere, and place it to `./libtensorflow`
	- https://github.com/PINTO0309/Tensorflow-bin/tree/master/C-library
- Build and Run

```
cmake ..
make -j4
./TensorFlowC 

pi@raspberrypi:~/aaa/build $ ./TensorFlowC 
Hello from TensorFlow C library version 2.0.0-beta1
89:  1.000
```
