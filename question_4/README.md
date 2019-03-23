## Question 4: CPU level parallelism and GPU
1. As a CV engineer, you should have had experience doing image manipulations using multi-threading in the CPU or using GPU kernels. First, study the single threaded program below that converts an RGB input that's loaded in an array to a grayscale image. The input array size is 1920x1080x3 flattened to an 1D array.
```c++
        int *get_greyscale(int* input) // this is the input image with a size of [image_width*image_height*3]
        {

        	int output_greyscale[image_width*image_height] = {0}; // output range from 0 - 255
        	for (int i=0; i<image_height; i++)
        	{
        		for (int j = 0; j<image_width; j++)
        		{
        			output_grey_scale = 0.2989 * input[i*image_width*3+j*3] + 0.5870 * input[i*image_width*3+j*3 + 1] + 0.1140 * input[i*image_width*3+j*3 + 2];
        		}
        	}
        	return output_greyscale;
        }
```
2. Rewrite the code above using CPU thread-level parallelism to utilize all the available cores on the CPU to accomplish the same output as above.
3. Write the pseudo-code in either OpenCL or CUDA for the kernel that performs the same function. Remember to specify what your threadgroup id and local thread id is. Write both the instantiation code in C++ as well as the kernel code in CUDA/OpenCL (pseudo code is okay). Assume a max thread count of 256 per thread group (or per block depending on your terminology), and thread group count (or block count) it has no limit.
4. Explain what you expect as the speed up is for each implementation. You do not need to be precise. Assume the following:
    - input image is 1080p (1920x1080) and 3 channels
    - GPU has a max 256 local cores, 8GB of local memory
    - CPU has 8 cores, runs at 2GHz, 8GB of RAM


### How to run the code.
Implemented #1 and #2 in python.
The following script will run the single threaded and multi-threaded function 5 times each and print out the average runtime.
It wil take less than 2 minutes to run this script.
```bash
python get_greyscale.py

Outputs -->
get_greyscale_single
repetition: 5 times
average time: 1.158 seconds

get_greyscale_multi_thread
repetition: 5 times
average time: 0.446 seconds # My machine has 4 cores
```
### Disclaimer for #4.3
My interactions with CUDA have mostly been through Pytorch. I'm always happy to learn new and useful tools, but I don't yet have direct experience writing CUDA or OpenCL. I will be writing this portion in C++ pseudocode as a result.

### 4.3 C++/CUDA Pseudocode
CUDA Host Code
```C++
void rgb_to_greyscale(const unsigned char* const inputImageHost, unsigned char* const inputImageGpu,
                      unsigned char* const inputImageCopyGpu, int imageWidth, int imageHeight)
{
    // We will first need to move the image to the GPU because this program will run on a CUDA device.
    // We'll need a copy of the original image and allocate memory for the final greyscale image.
    // Max thread count of 16 * 16 = 256
    int blockWidth = 16;
    int blockHeight = 16;
    int gridWidth = imageWidth / blockWidth;
    int gridHeight = imageHeight / blockHeight;

    // Round up the grid width and height
    if (gridWidth * blockWidth < imageWidth) gridWidth++;
    if (gridHeight * blockHeight < imageHeight) gridHeight++;

    const dim3 blockSize(blockWidth, blockHeight);
    const dim3 gridSize(gridWidth, gridHeight);

    // Copy image to the CUDA device and create another copy using device to device memcopy.
    CUDA_SAFE_CALL( cudaMemcpy(inputImageGpu, inputImageHost, sizeof(int) * MAX_DATA_SIZE, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(inputImageCopyGpu, inputImageGpu, sizeof(int) * MAX_DATA_SIZE, cudaMemcpyDeviceToDevice) );

    // Call the kernel that computes a greyscale image
    // rgb_to_greyscale_kernel(int width, int height, int *originalImage, int *greyscaleImage)
    rgb_to_greyscale_kernel<<<gridSize, blockSize>>>(imageWidth, imageHeight, inputImageGpu, inputImageCopyGpu);
    CUT_CHECK_ERROR("rgb_to_greyscale_kernel() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Copy the final greyscale image back to the host.
    CUDA_SAFE_CALL( cudaMemcpy(inputImageHost, inputImageCopyGpu, sizeof(int) * MAX_DATA_SIZE, cudaMemcpyDeviceToHost) );
}
```

CUDA Kernel Code
```C++
void __global__ rgb_to_greyscale_kernel(int width, int height, const unsigned char* const originalImage,
                                        unsigned char* const greyscaleImage)
{
    // Each thread on the GPU will process one pixel
    // Calculate the indices of the current pixel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the indices are valid
    if (i >= height || j >= width) return;

    int idx = i * img_width + j;
    int currentPixel = originalImage[idx];
    float red = currentPixel.x
    float green = currentPixel.y
    float blue = currentPixel.z
    greyscaleImage[idx] = .2989f * red + .587f * green + .114f * blue;
}
```

### Speed up for each implementation
1. Treating the RGB calculations as constant time, the single threaded program will take O(image_width * image_height) time.

2. 1 core has two threads, so a CPU with 8 cores will have 16 threads. Python cannot use the logical cores in parallel by default due to the global interpreter lock (GIL). I'm using numpy to work around python's GIL constraint. Numpy can work on all available threads because it's written in C. The multi-threaded program will be 16 times faster than the single threaded program. Time complexity wise, it is still O(image_width * image_height) time.

3. With CUDA, we can spawn 1 thread per pixel. Each thread will be responsible for getting the greyscale value per pixel. Because we are processing an image, it would be optimal if each thread block is 2 dimensional. 16x16 will let us use all 256 threads per block. There will be (1920x1080) / 256 = 8,100 blocks. Grid size will be set to 120x68. Assuming 256 local cores refer to CUDA cores, we can run 256 blocks in parallel. It will take `(8100 / 256) ~= 32` time to run this CUDA program. Image processing with CUDA will be `64,800 times ((1920 * 1080) / 32)` faster than the singled threaded program, and `4,050 times (((1920 * 1080) / 16) / 32)` faster than the multi-threaded program.
