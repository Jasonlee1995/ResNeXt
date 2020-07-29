# ResNeXt Implementation with Pytorch


## 0. Develop Environment
- Docker Image : tensorflow/tensorflow:1.13.2-gpu-py3-jupyter
- Pytorch : Stable (1.5) - Linux - Python - CUDA (10.2)
- Using Single GPU (not tested on cpu only)


## 1. Explain about Implementation


## 2. Brief Summary of *'Aggregated Residual Transformations for Deep Neural Networks'*

### 2.1. Goal
- Improve performance of image classification with low complexity

### 2.2. Intuition
- VGGNet & ResNet : stack layers strategy
  * Pros
    * Easy to design architecture
    * Reduce risk of over-adapting to a specific dataset due to simplicity of the rule
  * Cons
    * Reduce the free choices of hyperparameters
    * High complexity
- Inception : split-transform-merge strategy
  * Pros
    * Compelling accuracy with low complexity
  * Cons
    * Hard to design architecture
    * Many factors and hyperparameters to be designed

### 2.3. Dataset

### 2.4. ResNeXt Configurations

### 2.5. Classification Task
#### 2.5.1. Train  

#### 2.5.2. Test


## 3. Reference Paper
- Aggregated Residual Transformations for Deep Neural Networks [[paper]](https://arxiv.org/pdf/1611.05431.pdf)
