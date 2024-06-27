#include "GPURenderer.h"

#include "ErrorCheck.h"
#include "Kernels.h"
#include <string.h>

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
		m_Image = std::make_unique<Image>(width, height);
	}

	CU_CHECK(cudaGraphicsGLRegisterImage(
		&m_ImageCudaResource, m_Image->GetID(),
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard
	));
}

void GPURenderer::Render(Camera& camera)
{
	if (!m_Image)
		return;

	camera.RecalcMatrices();

	CU_CHECK(cudaGraphicsMapResources(1, &m_ImageCudaResource, (cudaStream_t)0));

	cudaArray* imgArray;
	CU_CHECK(cudaGraphicsSubResourceGetMappedArray(&imgArray, m_ImageCudaResource, 0, 0));

	cudaSurfaceObject_t surfObj;
	cudaResourceDesc surfObjResourceDesc;
	memset(&surfObjResourceDesc, 0, sizeof(surfObjResourceDesc));
	surfObjResourceDesc.resType = cudaResourceTypeArray;
	surfObjResourceDesc.res.array.array = imgArray;
	CU_CHECK(cudaCreateSurfaceObject(&surfObj, &surfObjResourceDesc));

	RenderParams params{
		.Surface = surfObj,
		.Camera = camera,
		.Width = m_Image->GetWidth(),
		.Height = m_Image->GetHeight()
	};

	traceRays(params);

	CU_CHECK(cudaGraphicsUnmapResources(1, &m_ImageCudaResource, (cudaStream_t)0));
}
