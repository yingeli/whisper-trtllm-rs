cmake_minimum_required(VERSION 3.1)

#set(TORCH_INCLUDE_DIRS "/opt/pytorch/pytorch/torch/csrc/api/include")
set(Torch_DIR "/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch")

set(TRTLLM_DIR "/app/tensorrt_llm")
set(TRTLLM_LIB_PATH "${TRTLLM_DIR}/lib/libtensorrt_llm.so")
set(TRTLLM_PLUGIN_PATH "${TRTLLM_DIR}/lib/libnvinfer_plugin_tensorrt_llm.so")
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/include")

set(TensorRT_LIBRARIES "/usr/local/tensorrt/include")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_VERBOSE_MAKEFILE 1)

# Define project name
project(whisper-trtllm)

find_package(Torch REQUIRED)

add_library(${PROJECT_NAME} SHARED "whisper.cpp")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${TRTLLM_LIB_PATH} ${TRTLLM_PLUGIN_PATH})

include_directories(${PROJECT_NAME} ${TORCH_INCLUDE_DIRS} ${TRTLLM_INCLUDE_DIR} ${TensorRT_LIBRARIES})