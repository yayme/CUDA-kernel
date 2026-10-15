from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sparse_attention",
    ext_modules=[
        CUDAExtension(
            name="sparse_attention",
            sources=[
                "csrc/sparse_attention_ext.cpp",
                "csrc/sparse_attention.cu",
            ],
            # -Xptxas -v prints registers per thread and spill bytes for every kernel at build time
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "-Xptxas", "-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
