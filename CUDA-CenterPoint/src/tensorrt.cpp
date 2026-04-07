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
#include <NvInferVersion.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <numeric>
#include <sstream>

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

    std::ostringstream oss;
    for(int i = 0; i < shape.nbDims; ++i){
        if(i) oss << " x ";
        // TensorRT 10 uses int64 dims; earlier versions used int32.
        oss << static_cast<long long>(shape.d[i]);
    }
    return oss.str();
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
    nvinfer1::IExecutionContext* context_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IRuntime* runtime_   = nullptr;

    virtual ~EngineImpl(){
#if NV_TENSORRT_MAJOR >= 10
        // TensorRT 10 removed destroy(); interfaces have virtual destructors.
        delete context_;
        delete engine_;
        delete runtime_;
#else
        if(context_) context_->destroy();
        if(engine_)  engine_->destroy();
        if(runtime_) runtime_->destroy();
#endif
    }

    bool load(const std::string& file){

        auto data = load_file(file);
        if(data.empty()){
            printf("Load engine %s failed.\n", file.c_str());
            return false;
        }

        runtime_ = nvinfer1::createInferRuntime(gLogger_);
        if(runtime_ == nullptr){
            printf("Failed to create runtime.\n");
            return false;
        }

#if NV_TENSORRT_MAJOR >= 10
        engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
#else
        engine_ = runtime_->deserializeCudaEngine(data.data(), data.size(), 0);
#endif
        if(engine_ == nullptr){
            printf("Failed to deserial CUDAEngine.\n");
            return false;
        }

        context_ = engine_->createExecutionContext();
        if(context_ == nullptr){
            printf("Failed to create execution context.\n");
            return false;
        }
        return true;
    }

    virtual int64_t getBindingNumel(const std::string& name) override{
#if NV_TENSORRT_MAJOR >= 10
        if(!context_) return 0;
        nvinfer1::Dims d = context_->getTensorShape(name.c_str());
        int64_t numel = 1;
        for(int i = 0; i < d.nbDims; ++i) numel *= static_cast<int64_t>(d.d[i]);
        return numel;
#else
        nvinfer1::Dims d = engine_->getBindingDimensions(engine_->getBindingIndex(name.c_str()));
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
#endif
    }

    virtual std::vector<int64_t> getBindingDims(const std::string& name) override{
#if NV_TENSORRT_MAJOR >= 10
        if(!context_) return {};
        nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
#else
        nvinfer1::Dims dims = engine_->getBindingDimensions(engine_->getBindingIndex(name.c_str()));
#endif
        std::vector<int64_t> output;
        output.reserve(dims.nbDims);
        for(int i = 0; i < dims.nbDims; ++i){
            output.push_back(static_cast<int64_t>(dims.d[i]));
        }
        return output;
    }

    virtual bool forward(const std::initializer_list<void*>& buffers, void* stream = nullptr) override{
#if NV_TENSORRT_MAJOR >= 10
        if(!context_ || !engine_) return false;

        const int nIO = engine_->getNbIOTensors();
        if(static_cast<int>(buffers.size()) != nIO){
            std::cerr << "TensorRT::forward buffers size mismatch. Expected " << nIO
                      << ", got " << buffers.size() << std::endl;
            return false;
        }

        std::vector<void*> bufs(buffers.begin(), buffers.end());
        for(int i = 0; i < nIO; ++i){
            const char* tname = engine_->getIOTensorName(i);
            if(!context_->setTensorAddress(tname, bufs[i])){
                std::cerr << "TensorRT::forward failed to setTensorAddress for " << tname << std::endl;
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

		printf("Engine %p detail\n", this);
#if NV_TENSORRT_MAJOR >= 10
        const int nIO = engine_->getNbIOTensors();
        int numInput = 0;
        int numOutput = 0;
        for(int i = 0; i < nIO; ++i){
            const char* tname = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(tname);
            if(mode == nvinfer1::TensorIOMode::kINPUT) numInput++;
            else if(mode == nvinfer1::TensorIOMode::kOUTPUT) numOutput++;
        }

        printf("Inputs: %d\n", numInput);
        int inIdx = 0;
        for(int i = 0; i < nIO; ++i){
            const char* tname = engine_->getIOTensorName(i);
            if(engine_->getTensorIOMode(tname) != nvinfer1::TensorIOMode::kINPUT) continue;
            printf("\t%d.%s : \tshape {%s}, %s\n",
                inIdx++,
                tname,
                format_shape(context_->getTensorShape(tname)).c_str(),
                data_type_string(engine_->getTensorDataType(tname))
            );
        }

        printf("Outputs: %d\n", numOutput);
        int outIdx = 0;
        for(int i = 0; i < nIO; ++i){
            const char* tname = engine_->getIOTensorName(i);
            if(engine_->getTensorIOMode(tname) != nvinfer1::TensorIOMode::kOUTPUT) continue;
            printf("\t%d.%s : \tshape {%s}, %s\n",
                outIdx++,
                tname,
                format_shape(context_->getTensorShape(tname)).c_str(),
                data_type_string(engine_->getTensorDataType(tname))
            );
        }
#else
        int numInput = 0;
        int numOutput = 0;
        for(int i = 0; i < engine_->getNbBindings(); ++i){
            if(engine_->bindingIsInput(i))
                numInput++;
            else
                numOutput++;
        }

        printf("Inputs: %d\n", numInput);
        for(int i = 0; i < numInput; ++i){
            int ibinding = i;
            printf("\t%d.%s : \tshape {%s}, %s\n",
                i,
                engine_->getBindingName(ibinding),
                format_shape(engine_->getBindingDimensions(ibinding)).c_str(),
                data_type_string(engine_->getBindingDataType(ibinding))
            );
        }

        printf("Outputs: %d\n", numOutput);
        for(int i = 0; i < numOutput; ++i){
            int ibinding = i + numInput;
            printf("\t%d.%s : \tshape {%s}, %s\n",
                i,
                engine_->getBindingName(ibinding),
                format_shape(engine_->getBindingDimensions(ibinding)).c_str(),
                data_type_string(engine_->getBindingDataType(ibinding))
            );
        }
#endif
    }
};

std::shared_ptr<Engine> load(const std::string& file){

    std::shared_ptr<EngineImpl> impl(new EngineImpl());
    if(!impl->load(file)) impl.reset();
    return impl;
}

}; // namespace TensorRT