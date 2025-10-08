#include "GPURenderer.h"

#include "ErrorCheck.h"
#include "Kernels.h"
#include <string.h>
#include <stdlib.h>

void GPURenderer::OnResize(uint32_t width, uint32_t height)
{
    // On program startup, this method could be called with 0 width and height,
    // which should be ignored
    if (width == 0 || height == 0)
        return;

    if (m_Image)
    {
        // Don't do anything if dimensions didn't change
        if (width == m_Image->GetWidth() && height == m_Image->GetHeight())
            return;

        CU_CHECK(cudaGraphicsUnregisterResource(m_ImageCudaResource));
        m_Image->Resize(width, height);
    }
    // Create image if it doesn't exist yet
    else
    {
        m_Image = std::make_unique<Image>(width, height, Image::Format::RGBA32F);
    }

    CU_CHECK(cudaGraphicsGLRegisterImage(
        &m_ImageCudaResource, m_Image->GetID(),
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone
    ));
}

static bool myisprint(unsigned char c) {
    return c >= 32 && c < 127;
}

static void hexdump(void* ptr, int buflen) {
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

void GPURenderer::Render(Scene& scene, Camera& camera, const RenderSettings& settings)
{
    if (!m_Image)
        return;

    // Clear screen and restart rendering if camera moved
    if (camera.RecalcMatrices())
        m_CurNumSamples = 0;

    Sphere* d_Spheres = nullptr;
    size_t spheresSize = scene.spheres.size() * sizeof(Sphere);
    CU_CHECK(cudaMalloc(&d_Spheres, spheresSize));
    CU_CHECK(cudaMemcpy(d_Spheres, scene.spheres.data(), spheresSize, cudaMemcpyHostToDevice));

    CU_CHECK(cudaGraphicsMapResources(1, &m_ImageCudaResource, (cudaStream_t)0));

    cudaArray* imgArray;
    CU_CHECK(cudaGraphicsSubResourceGetMappedArray(&imgArray, m_ImageCudaResource, 0, 0));

    cudaSurfaceObject_t surfObj;
    cudaResourceDesc surfObjResourceDesc;
    memset(&surfObjResourceDesc, 0, sizeof(surfObjResourceDesc));
    surfObjResourceDesc.resType = cudaResourceTypeArray;
    surfObjResourceDesc.res.array.array = imgArray;
    CU_CHECK(cudaCreateSurfaceObject(&surfObj, &surfObjResourceDesc));

    RenderKernelParams params{
        .renderParams = {
            .spheres = d_Spheres,
            .spheresCount = scene.spheres.size(),
            .camera = camera,
            .settings = settings
        },
        .surface = surfObj,		
        .width = m_Image->GetWidth(),
        .height = m_Image->GetHeight(),
        .curNumSamples = m_CurNumSamples
    };

    // TODO: remove
    //if (params.curNumSamples % 100 == 0)
    //{
    //    printf("=====================\HOST\n");
    //    for (int i = 0; i < params.renderParams.spheresCount; i++)
    //    {
    //        auto sphere = scene.spheres[i];
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

    render(params);

    CU_CHECK(cudaGraphicsUnmapResources(1, &m_ImageCudaResource, (cudaStream_t)0));

    CU_CHECK(cudaFree(d_Spheres));

    m_CurNumSamples++;
}
