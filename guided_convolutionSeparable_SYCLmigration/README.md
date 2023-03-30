# `convolutionSeparable` Sample

## Prior knowledge

- [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Beginner
- [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) - Beginner
- [SYCLomatic Manual](https://github.com/oneapi-src/SYCLomatic#syclomatic)

## Prerequisites

| Property              | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | SYCL compatible hardware 
| Software              | open source oneAPI DPC++/C++ Compiler

## Source code

- [CUDA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/convolutionSeparable) - Source code 
- [SYCL](https://github.com/vidyalatha-badde/convolutionSeparable/tree/master/guided_convolutionSeparable_SYCLmigration) - Migrated Code

## Purpose

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `dpct_output`              | Contains output of Intel® SYCLomatic Compatibility Tool which is fully migrated version of CUDA code.
| `sycl_migrated_optimized`            | Contains the optimized sycl code

## CUDA features demonstrated

This sample demonstrates the migration of the following prominent CUDA features: 

- Shared memory
- Constant memory
- Cooperative groups


## CUDA source code evaluation

A Separable Convolution is a process in which a single convolution can be divided into two or more convolutions to produce the same output. This sample implements a separable convolution filter of a 2D image with an arbitrary kernel. There are two functions in the code named convolutionRowsGPU and convolutionColumnsGPU in which the kernel functions (convolutionRowsKernel & convolutionColumnsKernel) are called where the loading of the input data and computations are performed. We validate the results with reference CPU separable convolution implementation by calculating the relative L2 norm.

## Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Tool assisted migration – SYCLomatic 

For this sample, the Intel SYCLomatic Compatibility tool automatically migrates ~100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/2_Concepts_and_Techniques/convolutionSeparable/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the Intel SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api
   ```
   
## Manual workarounds 
  
To find the device on which the code is getting executed replace the findCudaDevice (argc, (const char **) argv); with the following sycl get_device() API
 ```
 std::cout << "\nRunning on " << dpct::get_default_queue().get_device().get_info<sycl::info::device::name>() <<"\n";   
 ```

## Build the `convolutionSeparable` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `dpct_output` as well as `sycl_migrated_optimized` versions of the program.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `convolutionSeparable` Sample

### On Linux

You can run the programs for CPU and GPU. The commands indicate the device target.

1. Run `dpct_output` for CPU and GPU.
    ```
    make run_cpu
    make run_gpu
    ```
2. Run `sycl_migrated_optimized` for CPU and GPU.
    ```
    make run_cmo_cpu
    make run_cmo_gpu
    ```
    

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

