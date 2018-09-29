
Introduction
==============

Technology demonstration on how `galpy`_ orbit integration on Nvidia GPU via CUDA can be done by Python C extension which uses CUDA.
The purpose of this code is for demonstrating future possible galpy CUDA extensions.

This python package integrate orbits by Euler's method in solar system under a point mass Sun while mimic `galpy`_ package structure

**Work still in progress**

Installation
=================

Installation guide is provided below

Cross-platform checklist:
---------------------------------

- Compatible NVIDIA graphics card (`List of supported GPU`_)
- You have the latest NVIDIA driver and CUDA 10.0 installed as well as added to PATH (`Installation guide`_)
- You have Python >=3.6 installed, Anaconda is recommended (`Download Anaconda`_)
- You have downloaded this repository (``git clone https://github.com/henrysky/galpy_cuda_demo``)

.. _`List of supported GPU`: https://www.geforce.com/hardware/technology/cuda/supported-gpus
.. _`Installation guide`: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#introduction

Windows
--------

- You have supported Windows (`List of supported Windows`_)
- You have the latest Visual Studio 2017 installed with C/C++ toolset v14.14 or Visual Studio 2015

.. _List of supported Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements

Run Windows CMD under the respository, first we need to launch the right Visual compiler supported by CUDA 10.0

.. code-block:: bash

    %comspec% /k "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.14

Install the python package by

.. code-block:: bash

    python setup.py develop

To build wheels by

.. code-block:: bash

    python setup.py sdist bdist_wheel

Linux (Not tested yet)
----------------------------

- You have supported Linux (`List of supported Linux`_)
- You have supported GCC compiler (`List of supported GCC compiler`_ depending on your Linux distro)

Install the python package by

.. code-block:: bash

    python setup.py develop

To build wheels by

.. code-block:: bash

    python setup.py sdist bdist_wheel

.. _List of supported Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
.. _List of supported GCC compiler: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements

.. _`Download Anaconda`: https://www.anaconda.com/download/

Usage
=======

Unlike `galpy`_, ``Orbits`` here is an array of orbits and being integrated on GPU via CUDA in parallel.

.. code-block:: python

    from galpy_cuda_demo.orbits import Orbits
    import numpy as np
    import time

    num_obj = 10000  # total number of object to create and integrate
    x = np.random.normal(10, 2, num_obj)  # AU
    y = np.random.normal(10, 2, num_obj)  # AU
    vx = np.random.normal(3, 1, num_obj)  # AU/yr
    vy = np.random.normal(3, 1, num_obj)  # AU/yr

    # create cuda orbits and integrate
    o_cuda = Orbits(x=x, y=y, vx=vx, vy=vy, mode='cuda')
    start = time.time()
    o_cuda.integrate(steps=1000, dt=0.01)
    print('CUDA Time Spent: ', time.time() - start, 's')

    # create numpy cpu orbits and integrate
    o_cpu = Orbits(x=x, y=y, vx=vx, vy=vy, mode='cpu')
    start = time.time()
    o_cpu.integrate(steps=1000, dt=0.01)
    print('CPU Time Spent: ', time.time() - start, 's')

    print('CUDA orbits equal CPU orbits at 0.01% level: ', np.all(np.isclose(o_cpu.R, o_cuda.R)))

Computational Performance
==========================

The testing machines is Windows 10 RS5 x64, Anaconda 5.2 python 3.6.4 x64

As you can see NVIDIA GPU is only useful when you integrate a lot of objects because CUDA can integrate
millions of objects at each time steps in parallel, but can not integrate millions of time steps for single object
as expected (time steps must do sequentially).

- Integrating 100,000 objects with 1,000 time steps

    - NVIDIA GTX1060 6GB (GPU): ~ 1 seconds
    - Intel i7-7700K (CPU): ~ 17 seconds

- Integrating 1,000 objects with 100,000 time steps

    - NVIDIA GTX1060 6GB (GPU): ~ 17 seconds
    - Intel i7-7700K (CPU): ~ 17 seconds

- Integrating 1,000,000 objects with 1,000 time steps

    - NVIDIA GTX1060 6GB (GPU): ~ 8 seconds
    - Intel i7-7700K (CPU): ~ 250 seconds

To-Do list
=================

- Better Memory management on GPU
- Error Handling
- Accurancy of the Orbit Integration
- galpy's orbit integration/integrator and potential

Authors
=========
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Student, Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - *Project Supervisor* - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
.. _galpy: https://github.com/jobovy/galpy
