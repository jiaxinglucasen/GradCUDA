from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matmul_cuda_ext',
    ext_modules=[
        CUDAExtension(
            'matmul_cuda_ext',
            ['matmul_cuda_ext.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-Xcompiler', '/Zc:preprocessor']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)