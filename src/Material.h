#pragma once

#include <glm/glm.hpp>
#include <cuda/std/variant>
#include <cuda_runtime.h>

#include "Utils.h"
#include "HitInfo.h"
#include "Ray.h"

struct BasicMaterial
{
    glm::vec3 color{ 0.8f };
    glm::vec3 emissionColor{ 1.0f };
    float emissionStrength = 0.0f;
    float metallic = 0.0f;
    float fuzz = 0.0f;

    __host__ __device__
    bool ScatterRay(
        const Ray& rayIn,
        const HitInfo& hit,
        glm::vec3& absorption,
        glm::vec3& light,
        Ray& rayOut,
        uint32_t& seed
    ) const
    {
        if (randomFloat(seed) >= metallic)
        {
            // Diffuse
            glm::vec3 dir = hit.normal + randomUnitVec(seed);
            // Catch degenerate scatter direction
            if (nearZero(dir))
                dir = hit.normal;
            else
                dir = glm::normalize(dir);
            rayOut = { hit.position, dir };
        }
        else
        {
            // Metallic
            glm::vec3 reflected = glm::reflect(rayIn.dir, hit.normal);
            reflected += fuzz * randomUnitVec(seed);
            if (glm::dot(reflected, hit.normal) <= 0)
                return false;
            rayOut = { hit.position, glm::normalize(reflected) };
        }
        
        absorption = color;
        light = emissionColor * emissionStrength;
        return true;
    }
};

struct MirrorMaterial
{
    glm::vec3 tint{ 1.0f };

    __host__ __device__
        bool ScatterRay(
            const Ray& rayIn,
            const HitInfo& hit,
            glm::vec3& absorption,
            glm::vec3& light,
            Ray& rayOut,
            uint32_t& seed
        ) const
    {
        glm::vec3 reflected = glm::reflect(rayIn.dir, hit.normal);
        rayOut = { hit.position, reflected };
        absorption = tint;
        light = glm::vec3{ 0.0f };
        return true;
    }
};

class Material
{
public:
    // I really wanted to use cuda::std::variant here but I ran into weird
    // struct layout/size issues, so I'm falling back to a tagged union approach

    enum class Type : unsigned
    {
        // Should match the order of material types in the union
        Basic,
        Mirror
    };

#define MAT_TYPENAME(T) T##Material
#define MAT_UNION_NAME(T) m_As##T
#define MAT_UNION_DEF(T) MAT_TYPENAME(T) MAT_UNION_NAME(T)
#define MAT_METHODS(T) \
    Material(const MAT_TYPENAME(T)& m) : m_Type(Type::T), MAT_UNION_NAME(T)(m) {} \
    void Set(const MAT_TYPENAME(T)& m) { m_Type = Type::T; MAT_UNION_NAME(T) = m; }

    Material() : m_Type(Type::Basic), m_AsBasic(BasicMaterial{}) {}
    MAT_METHODS(Basic)
    MAT_METHODS(Mirror)

    __host__ __device__ Type GetType() const { return m_Type; }

    template<typename F>
    __host__ __device__ auto Visit(F f)
    {
        #define MAT_VISIT_CASE(T) case Type::T: return f(MAT_UNION_NAME(T))
        switch (m_Type)
        {
            MAT_VISIT_CASE(Basic);
            MAT_VISIT_CASE(Mirror);
        }
        #undef MAT_VISIT_CASE
    }

    template<typename F>
    __host__ __device__ auto Visit(F f) const
    {
        // TODO: check if this is a good way of providing a const visit method
        return const_cast<Material*>(this)->Visit(f);
    }

    __host__ __device__
        bool ScatterRay(
            const Ray& rayIn,
            const HitInfo& hit,
            glm::vec3& absorption,
            glm::vec3& light,
            Ray& rayOut,
            uint32_t& seed
        ) const
    {
        return Visit(
            [&](const auto& mat)
            {
                return mat.ScatterRay(
                    rayIn, hit, absorption, light, rayOut, seed
                );
            }
        );
    }

private:
    Type m_Type;
    union
    {
        MAT_UNION_DEF(Basic);
        MAT_UNION_DEF(Mirror);
    };

#undef MAT_METHODS
#undef MAT_UNION_DEF
#undef MAT_UNION_NAME
#undef MAT_TYPENAME
};


//class Material
//{
//public:
//    enum class Type : unsigned
//    {
//        // Should match the order of material types in the variant
//        Basic,
//        Mirror
//    };
//
//    __host__ __device__ Material() : m_Material(BasicMaterial{}) {}
//    template<typename M>
//    __host__ __device__ Material(M material) : m_Material(material) {}
//
//    template<typename M>
//    __host__ __device__ M& Get() { return cuda::std::get<M>(m_Material); }
//
//    template<typename M>
//    __host__ __device__ void Set(const M& mat) { m_Material = mat; }
//
//    __host__ __device__ Type GetType() const { return static_cast<Type>(m_Material.index()); }
//
//    template<typename F>
//    __host__ __device__ auto Visit(F func) { return cuda::std::visit(func, m_Material); }
//
//    __host__ __device__
//    bool ScatterRay(
//        const Ray& rayIn,
//        const HitInfo& hit,
//        glm::vec3& absorption,
//        glm::vec3& light,
//        Ray& rayOut,
//        uint32_t& seed
//    ) const
//    {
//        return cuda::std::visit(
//            [&](const auto& mat)
//            {
//                return mat.ScatterRay(
//                    rayIn, hit, absorption, light, rayOut, seed
//                );
//            },
//            m_Material
//        );
//    }
//
//private:
//    cuda::std::variant<
//        BasicMaterial,
//        MirrorMaterial
//    > m_Material;
//};
