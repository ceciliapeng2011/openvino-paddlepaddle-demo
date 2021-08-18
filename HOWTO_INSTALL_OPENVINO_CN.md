# 在本机上编译安装OpenVINO

## 1. 说明
OpenVINO对paddlepaddle的原生支持计划于**openvino 2022.1**版本发布。在这之前开发者可以从**github**上的**master**分支提前体验相关功能。跟之前对其它框架的支持不同的是，开发者可以直接将PaddlePaddle的模型传递给OpenVINO直接进行推理，而无需先用ModelOptimizer转换成IR（参考文档TBD）。

## 2. 从源码编译安装openvino

### 2.1 源码下载前的准备

若不能使用代理顺利的访问github服务的机器，建议将以下配置添加到本机~/.gitconfig

    [url "git://github.com"]
	      insteadOf="https://github.com"

### 2.2 源码获取

    git clone git@github.com:openvinotoolkit/openvino.git
    cd openvino
    export **OPENVINO_BASEDIR=`pwd` *#建议将openvino所在的目录加入系统环境变量，方便后续的安装运行*
    git submodule update --init --recursive
### 2.3 编译
    #配置
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DENABLE_MYRIAD=OFF \
    -DENABLE_VPU=OFF \
    -DENABLE_PYTHON=ON \
    -DNGRAPH_PYTHON_BUILD_ENABLE=ON \
    ..
    #编译，安装
    make -j$(nproc) ;make install
编译安装完成后，openvino将被安装在本机**${OPENVINO_BASEDIR}/openvino_dist**目录下。

