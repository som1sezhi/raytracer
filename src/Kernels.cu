#include "Kernels.h"

#include "ErrorCheck.h"
#include <stdio.h>
#include "Utils.h"

__device__
bool myisprint(unsigned char c) {
    return c >= 32 && c < 127;
}

__device__
void hexdump(void* ptr, int buflen) {
    unsigned char* buf = (unsigned char*)ptr;
    int i, j;
    for (i = 0; i < buflen; i += 16) {
        printf("%06x: ", i);
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%02x ", buf[i + j]);
            else
                printf("   ");
        printf(" ");
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%c", myisprint(buf[i + j]) ? buf[i + j] : '.');
        printf("\n");
    }
}

__global__
void renderKernel(RenderKernelParams params)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= params.width) || (y >= params.height))
        return;

    // TODO: remove
    //if (x == 0 && y == 0 && params.curNumSamples % 100 == 0)
    //{
    //    printf("=====================\nDEVICE\n");
    //    for (int i = 0; i < params.renderParams.spheresCount; i++)
    //    {
    //        auto sphere = params.renderParams.spheres[i];
    //        printf("center %f %f %f\n", sphere.center[0], sphere.center[1], sphere.center[2]);
    //        printf("radius %f\n", sphere.radius);
    //        printf("type %d\n", (int)sphere.material.GetType());
    //        printf("sizes %u %u %u %u\n", sizeof(sphere), sizeof(Sphere), sizeof(sphere.material), sizeof(Material));
    //        printf("offsets %u\n", offsetof(Sphere, material));
    //        //BasicMaterial& m = sphere.material.Get<BasicMaterial>();
    //        //printf("color %f %f %f\n", m.color[0], m.color[1], m.color[2]);
    //        hexdump(&sphere, sizeof(sphere));
    //        for (int i = 0; i < sizeof(sphere) / sizeof(float); i++)
    //        {
    //            printf("%f, ", *((float*)&sphere + i));
    //        }
    //        printf("\n\n");
    //    }
    //}

    int idx = x + params.width * y;
    uint32_t seed = idx * (params.curNumSamples + 1);
    Ray ray = params.renderParams.camera.CalcRay(
        x, y, params.renderParams.settings.antialias, seed
    );

    glm::vec3 color = getRayColor(ray, params.renderParams, seed);

    // Get previous pixel color
    float4 oldData;
    surf2Dread(&oldData, params.surface, x * 16, y);
    glm::vec3 old{ oldData.x, oldData.y, oldData.z };

    //  Accumulate color
    old = gammaToLinear(old);
    color = ((float)params.curNumSamples * old + color)
        / ((float)params.curNumSamples + 1);
    color = linearToGamma(color);

    // Write new color to surface
    float4 data = make_float4(color.r, color.g, color.b, 1.0f);
    surf2Dwrite(data, params.surface, x * 16, y);
}

void render(RenderKernelParams& kernelParams)
{
    const int tx = 8, ty = 8;
    dim3 blocks(kernelParams.width / tx + 1, kernelParams.height / ty + 1);
    dim3 threads(tx, ty);
    renderKernel<<<blocks, threads>>>(kernelParams);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
