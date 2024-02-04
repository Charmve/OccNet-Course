from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="bev_mmdet3d",
        version="0.1",
        packages=find_packages(),
        include_package_data=True,
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_v2_ext",
                module="ops.bev_pool_v2",
                sources=["src/bev_pool.cpp", "src/bev_pool_cuda.cu",],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="ops.iou3d",
                sources=["src/iou3d.cpp", "src/iou3d_kernel.cu",],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cpu.cpp",
                    "src/voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module="ops.roiaware_pool3d",
                sources=["src/roiaware_pool3d.cpp", "src/points_in_boxes_cpu.cpp",],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
