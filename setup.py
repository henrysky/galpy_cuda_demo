from Cython.Distutils import build_ext
from setuptools import setup, Extension, find_packages
import numpy
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


ext_modules = [Extension('galpy_cuda_demo.orbits.CUDAOrbits',
                         sources=['galpy_cuda_demo/orbits/CUDAOrbits.pyx'],
                         libraries=['CUDAOrbits'],
                         language='c',
                         extra_compile_args=[],
                         extra_link_args=[],
                         include_dirs=[numpy.get_include()]),
               ]

setup(
    name='galpy_cuda_demo',
    version='0.1.dev',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy'],
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://github.com/henrysky/galpy_cuda_demo',
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@mail.utoronto.ca',
    description='Orbit integration with CUDA technology demonstration',
    long_description=long_description,
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules)
