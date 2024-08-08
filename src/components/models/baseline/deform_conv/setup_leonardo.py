from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_flags = {}
gencodes = ['-arch=sm_75']

extra_compile_flags['nvcc'] = gencodes

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        extra_compile_args=extra_compile_flags),
    ],
    cmdclass={'build_ext': BuildExtension})
