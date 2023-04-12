# Deep learning application for solving differential equations
_authors: Chengze Shen, Senyu Tong_

In this project, we presented two deep learning approaches to solve first and
second order Poisson equations with Dirichlet boundary conditions.

## MLP solution
### Solving 2D-Laplace
For 2D-Laplace default (1 layer 100 nodes, batch size 32), run 
```
    python 2DLaplace.py 
```
also you can specify
* `-l`: # layer
* `-n`: # nodes,
* `-b`: batch size
* `-e`: # epochs
* `-d`: weight decay
* `-m`: momentum
#### Example
```
    python 2DLaplace.py -l 2 -n 64 -b 32 -e 200 -m 0.01 -d 9
```
  
### 3D Laplace
```
    python 3DLaplace.py -dim 3
```
Other parameters can be changed in the same way as in `2DLaplace.py`.
    
## GACNN solution
In addition to the naive MLP solutions to solving Laplace equations, we developed
a new approach, gradient approximation convolutional neural network (GACNN),
to apply the CNN architecture to solving differential equations. More specifically,
the discretized domain `D` is passed as input, and convolution operation is performed
on the region to mimic the extraction of gradients (within the small region).

The followings are instructions for running `GACNN.py` and `GACNN_naive.py` to
solve for Laplace equations. For more details, please refer to the
[report pdf](Deep_Learning_Applications_on_Solving_PDEs.pdf).

To run the code with default settings:
* discretized into 20 pieces,
* learning rate = 0.0001
* momentum = 0.5:
```
    python3 gacnn.py
```
The command will result in several graph generations (result and error at
certain epochs), as well as a running loss vs. epoch graph at the very end.

The same can be done to naive GACNN:
```
    python3 gacnn_naive.py
```

Prerequisites:
```
numpy
matplotlib
mpl_toolkits.mploy3d
pytorch
```
