# ml-math

## Module 1: The Input Tensor
Creating a function that takes in a raw data vector and scales it sothat the model can process it efficiently.

This is called __Normalization__. Normalization (or feauture scaling) is a preprocessing technique that rescales numerical input features to a standard range (typically 0 to 1 or -1 to 1) or distribution. We do this to ensure the models math won't be overwhelmed by the wildly different scales of the features. It keeps the importance of each feature balanced.

    Formula: V(norm) = v*(1/max(|v|))


## Module 2: Featue Redudancy:
Writing a script that checks what data is redundant.

This is checking for __Linear Independence__. If the span doesn't increase when a new feature is added, the feature is redundant. In ML,this is called Multicollinearity, and it makes the model weights highly unstable. 

__Multicollinearly__ is a statistical phenomenon in multiple regression analysis where two or more independent variables are highly linearly correlated, meaning one can be predicted from the others. 

    Formula: v1 = c*v2 (c is a constant scaler)


## Module 3: The Dense Layer
Implementing the forward pass where the input data is transformed by the model's weights.

This is the __Linear Transformation__. A neural network layer is simply a matrix that wraps and stretches the input space to find patterns.

    Formula: y = Wx
        $\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = x_1 \begin{bmatrix} w_{11} \\ w_{21} \end{bmatrix} + x_2 \begin{bmatrix} w_{12} \\ w_{22} \end{bmatrix}$

