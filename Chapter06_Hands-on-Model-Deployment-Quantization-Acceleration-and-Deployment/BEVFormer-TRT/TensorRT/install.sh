cd build
cmake .. -DCMAKE_TENSORRT_PATH=/usr/local/TensorRT
make -j$(nproc)
make install
