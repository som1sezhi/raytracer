#include "RayTracerApp.h"

int main(int, char**)
{
    AppSpec spec;
    spec.title = "Ray Tracing";
    spec.width = 1280;
    spec.height = 720;

    RayTracerApp app(spec);
    app.Run();

    return 0;
}
