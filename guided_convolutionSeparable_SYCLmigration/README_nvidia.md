# `convolutionSeparable` Sample

The `convolutionSeparable` sample is implemented using SYCL* by migrating code from original CUDA source code and offloading computations to a GPU/CPU.

| Property               | Description
|:---                    |:---
| What you will learn    | migrating CUDA to SYCL and optimizing it
| Time to complete       | 15 minutes

>**Note**: This sample is based on the [convolutionSeparable](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/convolutionSeparable) sample in the NVIDIA/cuda-samples GitHub repository.


## Purpose 

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `dpct_output`              | Contains output of Intel® SYCLomatic Compatibility Tool which is fully migrated version of CUDA code. 
| `convolutionSeparable_migrated_optimized`            | Contains the optimized sycl code 



## Prerequisites

| Property              | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | Tesla P100-PCIE-12GB
| Software              | Open source oneAPI DPC++ Compiler

## Key Implementation Details

This sample implements a separable convolution filter of a 2D image with an arbitrary kernel. There are two functions in the code named `convolutionRowsGPU` and `convolutionColumnsGPU` in which the kernel functions (`convolutionRowsKernel` & `convolutionColumnsKernel`) are called where the loading of the input data and computations are performed. We validate the results with reference CPU separable convolution implementation by calculating the relative L2 norm.

## Build and Run the convolutionSeparable Sample on NVIDIA GPU

To run convolutionSeparable SYCL migrated sample with CUDA NVIDIA backend we use open source DPC++ compiler. Refer [DPC++ LLVM(CLang-LLVM)](https://www.intel.com/content/www/us/en/developer/articles/technical/compiling-sycl-with-different-gpus.html#:~:text=their%20short%20descriptions.-,DPC%2B%2B%2DLLVM%20(CLang%2DLLVM),-The%20Data%20Parallel) to build and set-up the compiler.

**Steps**

Download the samples.

`git clone https://github.com/oneapi-src/oneAPI-samples.git`

Change to the sample directory.

To compile the sample following command is used,

`clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda convolutionSeparable.dp.cpp convolutionSeparable_gold.cpp main.cpp.dp.cpp -I../../../Common/` 

To Run

`./a.out`

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

