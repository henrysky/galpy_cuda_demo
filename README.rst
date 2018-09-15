
Introduction
==============

Technology demonstration on how `galpy`_-ish orbit integration on Nvidia GPU via CUDA can be done by Python C extension which uses CUDA.
The purpose of this code is for demonstrating future possible galpy CUDA extensions.

This python package integrate orbits by Euler's method in solar system under a point mass Sun while mimic `galpy`_ package structure

**Work still in progress**

Installation
=================

Installation should be easy on Windows and Linux supported by Nvidia

MacOS is not supported because Apple chosen AMD over Nvidia.

Windows
--------

Checklist:

- You have Windows 10 x64 installed with compatible Nvidia graphics card (700 series or above)
- You have the latest Nvidia driver and CUDA 9.2 installed as well as added to PATH
- You have Python >=3.6 installed, I recommend Anaconda
- You have Visual Studio 2017 Update 6 (toolset v14.13) installed with C/C++ support
- You have downloaded this repository

Run Windows CMD under the respository, first we need to launch the right Visual compiler supported by CUDA 9.2

.. code-block:: bash

    %comspec% /k "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.13

Then we compile the ``.cu`` files to ``.lib``

.. code-block:: bash

    nvcc -lib -o CUDAOrbits.lib galpy_cuda_demo/orbits/CUDAOrbits.cu

Build the python extension

.. code-block:: bash

    python setup.py build_ext -i -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include" -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64" --force clean

Install the python package by

.. code-block:: bash

    python setup.py develop

Linux (Not tested yet)
----------------------------

Checklist:

- You have reasonbly modern Linux installed with compatible Nvidia graphics card (700 series or above)
- You have the latest Nvidia driver and CUDA 9.2 installed as well as added to PATH
- You have Python >=3.6 installed, I recommend Anaconda
- You have C/C++ compiler installed
- You have downloaded this repository

Run Terminal under the respository, first we compile the ``.cu`` files to ``.a``

.. code-block:: bash

    nvcc -lib -o CUDAOrbits.a galpy_cuda_demo/orbits/CUDAOrbits.cu

Build the python extension

.. code-block:: bash

    python setup.py build_ext

Install the python package by

.. code-block:: bash

    python setup.py develop

Usage
=======

Unlike `galpy`_, ``Orbits`` here is an array of orbits and being integrated on GPU via CUDA in parallel.

.. code-block:: python

    from galpy_cuda_demo.orbits import Orbits
    import numpy as np
    import time

    num_obj = 10000  # total number of object to create and integrate
    x = np.random.normal(3, 0.1, num_obj)  # AU
    y = np.random.normal(3, 0.1, num_obj)  # AU
    vx = np.random.normal(20, 1, num_obj)  # AU/yr
    vy = np.random.normal(20, 1, num_obj)  # AU/yr

    # create cuda orbits and integrate
    o_cuda = Orbits(x=x, y=y, vx=vx, vy=vy, mode='cuda')
    start = time.time()
    o_cuda.integrate(steps=5000, dt=0.01)
    print('CUDA Time Spent: ', time.time() - start, 's')

    # create numpy cpu orbits and integrate
    o_cpu = Orbits(x=x, y=y, vx=vx, vy=vy, mode='cpu')
    start = time.time()
    o_cpu.integrate(steps=5000, dt=0.01)
    print('CPU Time Spent: ', time.time() - start, 's')

Performance Data
=================

Windows 10 x64, Anaconda 5.2 python 3.6

- Integrating 100,000 objects with 5,000 time steps

    - GTX1060 6GB: ~ 7 seconds
    - i7-7700K: ~ 150 seconds

- Integrating 10,000 objects with 5,000 time steps

    - GTX1060 6GB: ~ 0.5 seconds
    - i7-7700K: ~ 8 seconds

- Integrating 300,000 objects with 5,000 time steps

    - GTX1060 6GB: ~ 23 seconds
    - i7-7700K: ~ 520 seconds

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
