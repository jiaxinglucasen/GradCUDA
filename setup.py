# setup.py -- build script that compiles matmul_cuda_ext.cu into a Python module (a .pyd file on Windows).
# Run from an "x64 Native Tools Command Prompt for VS" with DISTUTILS_USE_SDK=1 set (see README).
from setuptools import setup                                        # Standard Python packaging entry point.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension   # PyTorch's helpers that call nvcc/cl.exe with the right include paths and flags.

setup(
    name='matmul_cuda_ext',                      # Package name; matches the module name Python imports.
    ext_modules=[                                # The compiled extension modules to build.
        CUDAExtension(
            'matmul_cuda_ext',                   # Name of the resulting module: `import matmul_cuda_ext`.
            ['matmul_cuda_ext.cu'],              # Source file(s) to compile.
            # Two compilers can be involved: cxx (plain C++ files) and nvcc (.cu files).
            # CUDA 13's headers require MSVC's standard-conforming preprocessor; nvcc passes the flag
            # on to cl.exe via -Xcompiler. (Putting it under 'cxx' does nothing for .cu files.)
            extra_compile_args={'cxx': [], 'nvcc': ['-Xcompiler', '/Zc:preprocessor']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension              # Use PyTorch's build step instead of the default, so .cu files compile.
    }
)
