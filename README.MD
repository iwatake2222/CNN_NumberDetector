# Sample projects to create deep learning applications
## How to start
```
git clone https://github.com/take-iwiw/CNN_NumberDetector.git
cd CNN_NumberDetector
git submodule init
git submodule update
```

## for C application
If you wan to build C/C++ application, you need header files.

```sh
cd 05_TensorflowLite_CPP/external_libs/tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
```

cmake build targets:

```
cd 05_TensorflowLite_CPP/
mkdir build && cd build
cmake ..		# PC Linux native
cmake .. -DBUILD_TARGET=RASPI_CROSS		# Cross compile for Raspberry Pi on PC Linux
cmake .. -DBUILD_TARGET=RASPI_NATIVE	# Native compile on Raspberry Pi
```
