from setuptools import setup,find_packages
import os
import glob
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def get_extensions():
    extensions = []

    ext_name = 'ivipcv._ext'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '8')
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available():
        define_macros += [('WITH_CUDA', None)]
        cuda_args = os.getenv('CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        op_files = glob.glob('./ivipcv/ops/csrc/pytorch/*')
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./ivipcv/ops/csrc/pytorch/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./ivipcv/ops/csrc')

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    setup(
        name="ivipcv",
        author='szc',
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )
