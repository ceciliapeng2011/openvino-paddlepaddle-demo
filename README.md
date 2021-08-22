# OpenVINO PaddlePaddle Integration Demo Preview

Note: The work presented in this repository is for demo and preview purposes only. 

This repository provides a set of sample code that demostrate how to run PaddlePaddle models in OpenVINO. 

![scrnli_8_12_2021_7-58-54 PM](https://user-images.githubusercontent.com/1720147/129298808-b084d7fb-9585-404b-95f9-c4346c21da6b.png)

## How to Setup

### Step 0 - Clone the repository 

Download this OpenVINO PaddlePaddle sample repository! Please note that we will install materials inside the openvino-paddlepaddle-demo folder as the default working directory.  
```
git clone https://github.com/raymondlo84/openvino-paddlepaddle-demo.git
cd openvino-paddlepaddle-demo
```

### Step 1 - Install OpenVINO from source
We install the OpenVINO source library to the openvino/openvino_dev directory by default. 

Download the OpenVINO Source
```
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
```

Install the dependencies for OpenVINO source and Python
- For Linux
```
chmod +x install_build_dependencies.sh
./install_build_dependencies.sh
pip install -r inference-engine/ie_bridges/python/src/requirements-dev.txt
```

- For Mac
```
#install Python Dependencies
pip install -r inference-engine/ie_bridges/python/src/requirements-dev.txt
```

Compile the source code with Python option enabled.

```
OPENVINO_BASEDIR = `pwd`
mkdir build
cd build
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
-DPYTHON_EXECUTABLE=$(which python3) \
-DENABLE_MYRIAD=OFF \
-DENABLE_VPU=OFF \
-DENABLE_PYTHON=ON \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
..

make -j$(nproc); make install
```

### Step 2 - Setup the OpenVINO PaddlePaddle Sample from GitHub
Create the virtual environment and install dependencies. Now let's start from the working directory openvino-paddlepaddle-demo. 

```sh
cd openvino-paddlepaddle-demo
python3 -m venv openvino_env
source openvino_env/bin/activate

#install the dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

#install the kernel to Jupyter
python -m ipykernel install --user --name openvino_env
```

### Step 3 - Setup PaddleDetection and Dependencies
Note: please make sure you are in the openvino-paddlepaddle-demo directory.
```sh
git clone https://github.com/PaddlePaddle/PaddleDetection.git
pip install --upgrade -r PaddleDetection/requirements.txt -i https://mirror.baidu.com/pypi/simple
#Optional
#python PaddeDetection/setup.py install
```

### Step 4 - Execute the Jupyter Notebooks
Enable the virtual environment, and also the OpenVINO environment. Then, execute the jupyter lab.   
```sh 
#For Linux and Mac
source openvino_env
source openvino/openvino_dist/bin/setupvars.sh
cd openvino-paddlepaddle-demo
jupyter lab notebooks
```

Note: Please make sure you select the openvino_env as the Kernel in the Notebooks.

### References:
- [Compiling OpenVINO from Source](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode)
- [Converting a Paddle* Model]( https://github.com/openvinotoolkit/openvino/blob/35e6c51fc0871bade7a2c039a19d8f5af9a5ea9e/docs/MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md)

### Notes and Disclaimers
* Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](http://www.Intel.com/PerformanceIndex).
* Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  
* No product or component can be absolutely secure. 
* Intel technologies may require enabled hardware, software or service activation.
* Your costs and results may vary. 
* © Intel Corporation.  Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others. 
