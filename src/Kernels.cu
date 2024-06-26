#include "Kernels.h"

#include "ErrorCheck.h"

__global__
void traceRaysKernel(cudaSurfaceObject_t surf, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;

    unsigned char r = (unsigned char)(255.99 * (float)x / width);
    unsigned char g = (unsigned char)(255.99 * (float)y / height);
    uchar4 data = make_uchar4(r, g, 255, 255);
    surf2Dwrite(data, surf, x * 4, y);
}

void traceRays(cudaSurfaceObject_t surfObj, uint32_t width, uint32_t height)
{
    const int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    traceRaysKernel<<<blocks, threads>>>(surfObj, (int)width, (int)height);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
