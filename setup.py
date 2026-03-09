from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension;

setup(
    name= "sparse_attention",
    ext_modules =[
        CUDAExtension(
            name="sparse_attention",
            sources = [
                "sparse_attention_ext.cpp",
                "sparse_attention.cu"
            ]
        )
    ],
    cmdclass ={"build_ext": BuildExtension}
    
)
