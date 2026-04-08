/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "tensorrt.hpp"
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <numeric>
#include <memory>
#include <unordered_map>

namespace TensorRT{

static class Logger : public nvinfer1::ILogger {
    public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR){
            std::cerr << "[NVINFER LOG]: " << msg << std::endl;
        }
    }
}gLogger_;

static std::string format_shape(const nvinfer1::Dims& shape){

    char buf[200] = {0};
    char* p = buf;
    for(int i = 0; i < shape.nbDims; ++i){
        if(i + 1 < shape.nbDims)
            p += sprintf(p, "%d x ", static_cast<int>(shape.d[i]));
        else
            p += sprintf(p, "%d", static_cast<int>(shape.d[i]));
    }
    return buf;
}

static std::vector<uint8_t> load_file(const std::string& file){

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

static const char* data_type_string(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT: return "Float32";
        case nvinfer1::DataType::kHALF: return "Float16";
        case nvinfer1::DataType::kINT32: return "Int32";
        // case nvinfer1::DataType::kUINT8: return "UInt8";
        case nvinfer1::DataType::kINT8: return "Int8";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        default: return "Unknow";
    }
}

class EngineImpl : public Engine{
public:
    template<typename T>
    static void destroy_pointer(T* ptr){
        if(!ptr) return;
#if NV_TENSORRT_MAJOR >= 10
        delete ptr;
#else
        ptr->destroy();
#endif
    }

    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::unordered_map<std::string, int> binding_name_to_index_;

    virtual ~EngineImpl() = default;

    bool load(const std::string& file){

        auto data = load_file(file);
        if(data.empty()){
            printf("Load engine %s failed.\n", file.c_str());
            return false;
        }

        runtime_.reset(nvinfer1::createInferRuntime(gLogger_), destroy_pointer<nvinfer1::IRuntime>);
        if(runtime_ == nullptr){
            printf("Failed to create runtime.\n");
            return false;
        }

        #if NV_TENSORRT_MAJOR >= 10
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()), destroy_pointer<nvinfer1::ICudaEngine>);
        #else
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size(), nullptr), destroy_pointer<nvinfer1::ICudaEngine>);
        #endif
        if(engine_ == nullptr){
            printf("Failed to deserial CUDAEngine.\n");
            return false;
        }

        context_.reset(engine_->createExecutionContext(), destroy_pointer<nvinfer1::IExecutionContext>);
        if(context_ == nullptr){
            printf("Failed to create execution context.\n");
            return false;
        }

#if NV_TENSORRT_MAJOR >= 10
        binding_name_to_index_.clear();
        for(int i = 0; i < engine_->getNbIOTensors(); ++i){
            binding_name_to_index_[engine_->getIOTensorName(i)] = i;
        }
#endif
        return true;
    }

    virtual int64_t getBindingNumel(const std::string& name) override{
        nvinfer1::Dims d = getBindingDimensions(name);
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    virtual std::vector<int64_t> getBindingDims(const std::string& name) override{
        nvinfer1::Dims dims = getBindingDimensions(name);
        std::vector<int64_t> output(dims.nbDims);
        std::transform(dims.d, dims.d + dims.nbDims, output.begin(), [](int32_t v){return v;});
        return output;
    }

    virtual bool forward(std::initializer_list<void*> buffers, void* stream = nullptr) override{
#if NV_TENSORRT_MAJOR >= 10
        if(static_cast<int>(buffers.size()) != engine_->getNbIOTensors()){
            printf("TensorRT binding count mismatch, expected %d, got %zu.\n", engine_->getNbIOTensors(), buffers.size());
            return false;
        }

        auto it = buffers.begin();
        for(int i = 0; i < engine_->getNbIOTensors(); ++i, ++it){
            const char* tensor_name = engine_->getIOTensorName(i);
            if(!context_->setTensorAddress(tensor_name, *it)){
                printf("Failed to set tensor address for %s.\n", tensor_name);
                return false;
            }
        }
        return context_->enqueueV3((cudaStream_t)stream);
#else
        return context_->enqueueV2(buffers.begin(), (cudaStream_t)stream, nullptr);
#endif
    }

    virtual void print() override{

        if(!context_){
			printf("Infer print, nullptr.\n");
			return;
		}

        int numInput = 0;
        int numOutput = 0;
        for(int i = 0; i < numBindings(); ++i){
            if(isInput(i))
                numInput++;
            else
                numOutput++;
        }

		printf("Engine %p detail\n", this);
		printf("Inputs: %d\n", numInput);
		for(int i = 0; i < numInput; ++i){
            int ibinding = i;
            const char* binding_name = bindingName(ibinding);
			printf("\t%d.%s : \tshape {%s}, %s\n",
                i,
                binding_name,
                format_shape(getBindingDimensions(binding_name)).c_str(),
                data_type_string(getBindingDataType(binding_name))
            );
		}

		printf("Outputs: %d\n", numOutput);
		for(int i = 0; i < numOutput; ++i){
			int ibinding = i + numInput;
            const char* binding_name = bindingName(ibinding);
			printf("\t%d.%s : \tshape {%s}, %s\n",
                i,
                binding_name,
                format_shape(getBindingDimensions(binding_name)).c_str(),
                data_type_string(getBindingDataType(binding_name))
            );
		}
    }

private:
    int numBindings() const{
#if NV_TENSORRT_MAJOR >= 10
        return engine_->getNbIOTensors();
#else
        return engine_->getNbBindings();
#endif
    }

    const char* bindingName(int index) const{
#if NV_TENSORRT_MAJOR >= 10
        return engine_->getIOTensorName(index);
#else
        return engine_->getBindingName(index);
#endif
    }

    bool isInput(int index) const{
#if NV_TENSORRT_MAJOR >= 10
        return engine_->getTensorIOMode(bindingName(index)) == nvinfer1::TensorIOMode::kINPUT;
#else
        return engine_->bindingIsInput(index);
#endif
    }

    int bindingIndex(const std::string& name) const{
#if NV_TENSORRT_MAJOR >= 10
        auto it = binding_name_to_index_.find(name);
        return it == binding_name_to_index_.end() ? -1 : it->second;
#else
        return engine_->getBindingIndex(name.c_str());
#endif
    }

    nvinfer1::Dims getBindingDimensions(const std::string& name) const{
#if NV_TENSORRT_MAJOR >= 10
        return context_->getTensorShape(name.c_str());
#else
        return engine_->getBindingDimensions(bindingIndex(name));
#endif
    }

    nvinfer1::DataType getBindingDataType(const std::string& name) const{
#if NV_TENSORRT_MAJOR >= 10
        return engine_->getTensorDataType(name.c_str());
#else
        return engine_->getBindingDataType(bindingIndex(name));
#endif
    }
};

std::shared_ptr<Engine> load(const std::string& file){

    std::shared_ptr<EngineImpl> impl(new EngineImpl());
    if(!impl->load(file)) impl.reset();
    return impl;
}

}; // namespace TensorRT