#include "RayTracerApp.h"

int main(int, char**)
{
    AppSpec spec;
    spec.Title = "cool gpu-rendered gradient";
    spec.Width = 1280;
    spec.Height = 720;

    RayTracerApp app(spec);
    app.Run();

    return 0;
}
