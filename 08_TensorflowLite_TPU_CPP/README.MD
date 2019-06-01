
```
# Install libraries needed
sudo apt install -y curl wget cmake
sudo apt install -y libc6-dev libc++-dev libc++abi-dev
sudo apt install -y libusb-1.0-0 


# Clone and download code
cd ~/
git clone https://github.com/iwatake2222/CNN_NumberDetector.git
cd CNN_NumberDetector/08_TensorflowLite_TPU_CPP
git submodule init
git submodule update
cd external_libs/edgetpu-native/
git submodule init
git submodule update
cd tensorflow/tensorflow/lite/tools/make/
bash ./download_dependencies.sh


# Build code 
cd ~/CNN_NumberDetector/08_TensorflowLite_TPU_CPP/
mkdir build && cd build

## Download model and sample input image file
wget https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
wget http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xfz mobilenet_v2_1.0_224_quant.tgz
wget https://coral.withgoogle.com/static/docs/images/parrot.jpg

## Copy library
cp ../external_libs/edgetpu-native/libedgetpu/libedgetpu_arm64.so libedgetpu.so.1

## Build and Run
cmake .. -DBUILD_TARGET=JETSON_NATIVE
make -j2
sudo LD_LIBRARY_PATH=./ ./NumberDetector
```
