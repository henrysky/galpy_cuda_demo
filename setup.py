import os
import subprocess
import sys

import numpy
from Cython.Distutils import build_ext
from setuptools import setup, Extension, find_packages

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# nvcc compile first
nvcc_code = subprocess.call(
    f"nvcc -lib -o CUDAOrbits.{'lib' if sys.platform.startswith('win') else 'a'} galpy_cuda_demo/orbits/CUDAOrbits.cu",
    shell=sys.platform.startswith('win'))

# check to make sure nvcc did successfully compiled the CUDA code
if nvcc_code != 0:
    raise OSError("NVCC compilation failed")

ext_modules = [Extension('galpy_cuda_demo.orbits.CUDAOrbits',
                         libraries=["CUDAOrbits", "cudart"],
                         language="c",
                         extra_compile_args=[],
                         extra_link_args=[],
                         # hardcode x64 because CUDA will drop x86 support anyway
                         library_dirs=[os.path.join(os.getenv("CUDA_PATH"), "lib", "x64")],
                         include_dirs=[numpy.get_include(), os.path.join(os.getenv("CUDA_PATH"), "include")],
                         sources=['galpy_cuda_demo/orbits/CUDAOrbits.pyx'],
                         ),
               ]

setup(
    name='galpy_cuda_demo',
    version='0.1.dev',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'],
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://github.com/henrysky/galpy_cuda_demo',
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@mail.utoronto.ca',
    description='Technology demonstration of orbits integration with CUDA for galpy',
    long_description=long_description,
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules)
