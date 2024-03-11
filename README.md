# Adaptive Spiral
This repo contains the implementation for the ICCV23 paper "Adaptive Spiral Layers for Efficient 3D Representation Learning on Meshes".

## The Adaptive Spiral Operator
Adaptive Spiral Convolution dynamically adjusts the length of the spiral trajectory and the parameters of the transformation for each processed vertex and
mesh. 
<p align="center"><img src="./assets/142507271/35d7db00-8519-4719-a3b3-e321b11a14b3" align=middle width=77.33054999999999pt height=13.156093499999999pt/></p>
You can find its implementation in adaptive_spiral.py.

## Installation
In our implementation we follow the original [SpiralNet++] (https://github.com/sw-gong/spiralnet_plus/tree/master) codebase.
The code is developed using Python 3.6 on Ubuntu 16.04. The models were trained and tested with NVIDIA Tesla V100.
* [Pytorch](https://pytorch.org/) (1.13.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (2.3.1)
* [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/openmesh-python) (1.1.3)
* [MPI-IS Mesh](https://github.com/MPI-IS/mesh): installed from source.

## Train 3D Face Reconstruction on CoMA
Download the CoMA dataset from [here] (https://coma.is.tue.mpg.de/). Run the interpolation experiment :
```
python -m reconstruction.main --config ./config/reconstruction/AdaptiveSpiralconv.yaml
```

## Citation
Please consider citing our work:
```
@inproceedings{babiloni2023adaptive,
  title={Adaptive Spiral Layers for Efficient 3D Representation Learning on Meshes},
  author={Babiloni, Francesca and Maggioni, Matteo and Tanay, Thomas and Deng, Jiankang and Leonardis, Ales and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14620--14631},
  year={2023}
}
