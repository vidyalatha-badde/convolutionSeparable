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
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

This sample implements a separable convolution filter of a 2D image with an arbitrary kernel. There are two functions in the code named `convolutionRowsGPU` and `convolutionColumnsGPU` in which the kernel functions (`convolutionRowsKernel` & `convolutionColumnsKernel`) are called where the loading of the input data and computations are performed. We validate the results with reference CPU separable convolution implementation by calculating the relative L2 norm.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

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

   By default, this command sequence will build the `dpct_output` as well as `convolutionSeparable_migrated_optimized` versions of the program.

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
2. Run `convolutionSeparable_migrated_optimized` for CPU and GPU.
    ```
    make run_cmo_cpu
    make run_cmo_gpu
    ```
### Build and Run the `convolutionSeparable` Sample in Intel® DevCloud

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`.

1. Open a terminal on a Linux* system.
2. Log in to Intel® DevCloud.
    ```
    ssh devcloud
    ```
3. Download the samples.
    ```
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
4. Change to the sample directory.
5. Configure the sample for a GPU node.
   ```
   qsub  -I  -l nodes=1:gpu:ppn=2 -d .
   ```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

     |Available Nodes  |Command Options
     |:---             |:---
     | GPU	         |`qsub -l nodes=1:gpu:ppn=2 -d .`
     | CPU	         |`qsub -l nodes=1:xeon:ppn=2 -d .`

6. Perform build steps as you would on Linux.
7. Run the programs.
8. Clean up the project files.
    ```
    make clean
    ```
9. Disconnect from the Intel® DevCloud.
    ```
    exit
    ```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

