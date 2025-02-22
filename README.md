# whisper-trt

python3 -m pip config set global.break-system-packages true

sudo apt-get install clang libc++-dev libc++abi-dev

python3 ./scripts/build_wheel.py --cuda_architectures "75-real" --trt_root /usr/local/tensorrt --extra-cmake-vars ENABLE_MULTI_DEVICE=0 --cpp_only --clean

python3 ./scripts/build_wheel.py --cuda_architectures native --trt_root /usr/local/tensorrt --cpp_only --clean


sudo apt install libopenmpi-dev

export LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH_USE_PYTORCH=1

export AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
  --name mystorageaccount --query connectionString -o 

az storage blob upload-batch \
  --source /path/to/your/directory \
  --destination mycontainer

  