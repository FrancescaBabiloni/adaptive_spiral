# Adaptive Spiral
This repo contains the implementation for the ICCV23 paper "Adaptive Spiral Layers for Efficient 3D Representation Learning on Meshes".

## The Adaptive Spiral Operator

![adaptive_spiral](https://github.com/Fb2221/adaptive_spiral/assets/142507271/87429db2-6ebb-439e-a1a2-0ed7e37ca146)

Adaptive Spiral Convolution extracts and transforms the vertices in the 3D mesh following a spiral
order. It then dynamically adjusts the length of the spiral trajectory and the parameters of the transformation for each processed vertex and
mesh. 

You can find its implementation in adaptive_spiral.py.

## Installation
The code is developed using Python 3.6 on Ubuntu 16.04. The models were trained and tested with NVIDIA Tesla V100.
* [Pytorch](https://pytorch.org/) (1.13.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (2.3.1)
* [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/openmesh-python) (1.1.3)
* [MPI-IS Mesh](https://github.com/MPI-IS/mesh): installed from source.

## Train 3D Face Reconstruction on CoMA
We follow the original [SpiralNet++](https://github.com/sw-gong/spiralnet_plus/tree/master) codebase.
To perform a quick run, download the CoMA dataset from [here](https://coma.is.tue.mpg.de/) and then run the interpolation experiment using:
```
python -m reconstruction.main --config ./config/reconstruction/AdaptiveSpiralconv.yaml
```

## Citation
If you find this repo useful consider citing our work:
```
@inproceedings{babiloni2023adaptive,
  title={Adaptive Spiral Layers for Efficient 3D Representation Learning on Meshes},
  author={Babiloni, Francesca and Maggioni, Matteo and Tanay, Thomas and Deng, Jiankang and Leonardis, Ales and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14620--14631},
  year={2023}
}
