# Linear Algebra for Machine Learning

## Segment 1: Data Structures for Algebra
### Libraries used:

#### numpy:
Numpy is used for scientific computing in Python. This library provides multidimensional array objects, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operation on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

```np.linspace(start, end, n-points)```: This function is used to create a one-dimensional array of evenly spaced numbers over a specified interval.

#### matplotlib:
This library is used to create static, animated, and interactive visualizations. We are using this to draw graphs in machine learning.

```figure, axes = plt.sublots()```: Subplots are individual plots arranged in a grid within a single overall canvas.
- Figure: Canvas that contains the plots.
- Axes: Plot area, where the x and y axes are, with titles and labels. 

#### torch:
PyTorch tensors are what we use to train and deploy machine learning models. They are designed to be pythonic, meaning they feel and behave like NumPy arrays. The advantage to PyTorch tensors is that they can be easily used for operations on GPU's. This means we can have matrix operations running in parallel.

```variable = torch.tensor(25, dtype = torch.float16)```: This is used to specify the data-type of the tensor data.

#### tensorflow:
With TensorFlow the tensors are created with wrappers.
- ```tf.Variable```
- ```tf.constant```
- ```tf.placeholder```
- ```tf.SparseTensor```

### Tensors:
Tensors are specific to machine learning. They are a generalization of vectors and matrices with any number of dimensions.
#### Some Tensor Examples:
- __Scalar__ : x (no dimensions)
- __Vector__ : [x x x] (1 dimensional - linear representation of a collection of scalars)
- __Matrix__ : (2 dimensional)

![Alt text for the image](https://helpingwithmath.com/wp-content/uploads/2021/10/Matrix-Notation.jpg)

- __3-Tensor__ : (3 dimensional)

| Dimensions | Mathematical Name | Description |
|---|---|---|
| 0 | scalar | magnitude only |
| 1 | vector | array |
| 2 | matrix | flat table |
| 3 | 3-tensor | 3D table |
| n | n-tensor | higher dimensional |


### Scalars:
- No Dimensions
- Singular Number
- Has a data-type (int, float)

### Vectors:
- One-dimensional array of numbers
- Arranged in order, so each element can be accessed by its index
- Vectors represent a point in space.

![Alt text for the image](https://mathtec.weebly.com/uploads/2/9/0/5/29050183/vector9_orig.jpg)

### Vector Transposition:
Vector transposition is when you convert a vectors rows into its columns or visa versa. So why do we do vector transposition?
We use machine learning is used to change a vector's orientation between row and column shapes to do matrix multiplication. We do this beacuse matrix multiplication has strict dimensional requirements. 

__Matrix Multiplication:__
- Dot products
- Linear transformations
- Gradient Descent & Loss functions

![Alt text for the image](https://peerherholz.github.io/Cog_Com_Neuro_ML_DL/_images/transpose.png)

### Zero Vectors
Vectors with magnitude of 0 and no specific direction. These will have no effect when added to another vector.

### Norms
First we need to understand that vectors represent a magnitude and direction from the origin. Norms are functions that quantify vector magnitude. 

__L2 Norm:__
The most common and most important norm function is the L2 norm function.

![Alt text for the image](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/vector-norms-12.png)

Measures the straight-line/simple distance from the origin to a point in Euclidean space.

__L1 Norm:__
This is also a pretty common norm in machine learning.

![Alt text for the image](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/vector-norms-9.png)

__Squared L2 Norm:__
This is like L2 norm, but you dont have to use the square root. This makes the squared L2 norm function computationally cheaper to use than the L2 norm, because:
- Squared L2 norm equals transpose of x times x
- The derivative of element x requires that element alone. Whereas L2 norm requires the entire vector.
The downside of squared L2 norm is that it grows slowly near the origin, so it cant be used if you need to distinguish between zero and near-zero.

__Max Norm:__
This occurs reasonably frequently in machine learning. 

![Alt text for the image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdIPzbKT6iyumXzzCcm1frqFFa6S7AL7T8iA&s)

Returns the absolute value of the largest-magnitude element in the vector.

All of the norms above are specific cases or variations of the 
__Generalized Lp Norm:__

![Alt text for the image](https://miro.medium.com/v2/resize:fit:732/1*CUeKs_0Sfq4xVcn_jfDMsw.png)

- p must be:
    - A real number
    - Greater than or equal to 1
- Can derive L1, L2 and Lx norm formulae by substituting p.
- Norms, particularly L1 and L2, used to requlize objective functions.

__These normalization methods have different use cases:__
- L1 norm: This sums the absolute differences of each value within a given vector. Promoting sparsity and robustness to outliers. This makes it ideal for feature selection. In other words, it is used whenever the difference between zero and non-zero is key.
- L2 norm: This is the square root of the sums of squares for each value within a given vector. This scales down the outliers of the vector.  

### Unit Vectors:
This is a special case of vectors where its length is equal to 1. Technically ||x|| = 1

![Alt text for the image](https://media.geeksforgeeks.org/wp-content/uploads/20220308175724/vectoranditsunitvectordiagram-660x351.jpg)

## Author:
Junior Developer: Jared Van Eeden
- LinkedIn: https://www.linkedin.com/in/jared-van-eeden-483499258/
- Personal Portfolio: https://personal-portfolio-bf19d.web.app/index.html
