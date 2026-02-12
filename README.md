# ml-math

## Module 1: The Input Tensor
Creating a function that takes in a raw data vector and scales it sothat the model can process it efficiently.

This is called __Normalization__. Normalization (or feauture scaling) is a preprocessing technique that rescales numerical input features to a standard range (typically 0 to 1 or -1 to 1) or distribution. We do this to ensure the models math won't be overwhelmed by the wildly different scales of the features. It keeps the importance of each feature balanced.

    Formula: V(norm) = v*(1/max(|v|))


## Module 2: Featue Redudancy:
Writing a script that checks what data is redundant.

This is checking for __Linear Independence__. If the span doesn't increase when a new feature is added, the feature is redundant. In ML,this is called Multicollinearity, and it makes the model weights highly unstable. 

__Multicollinearly__ is a statistical phenomenon in multiple regression analysis where two or more independent variables are highly linearly correlated, meaning one can be predicted from the others. 

    Formula: v(1) = c*v(2) (c is a constant scaler)


## Module 3: The Dense Layer
Implementing the forward pass where the input data is transformed by the model's weights.

This is the __Linear Transformation__. A neural network layer is simply a matrix that wraps and stretches the input space to find patterns.

    Formula: y = Wx


## Module 4: Deep Learning
Combining two layers into one single operation

This is __Matrix Composition__. Applying layer 1 and then layer 2 is the same as applying their product. This is why we can stack dozens of layers in deep learning to perform complex data warping.

    Formula: W(total) = W(2) * W(1)


## Module 5: Information Stability
Creating a safety check to ensure the model isn't deleting data.

The __Determinant__ measure the factor by which area or volume is scaled. 
- Determinant > 1 : Space is expanding. (Information is being added)
- 0 < Determinant < 1 : Space is contracting. (Information is being lost)
- Determinant = 0 : Space has collapsed into a lower dimension. (All information is lost)
- Determinant < 0 : Space has been flipped.


## Module 6: Ideal State Solver
Given a target result, find the exact input required to produce it. 

This uses the __Inverse Matrix__ to play the transformation in reverse. We usually use Gradient Descent, but calculating the inverse is the foundation for solving __Linear Systems__ and finding perfect weights in models like __Linear Reression__.

    Formula: x = W**-1y


## Module 7: Dimension Compression
Projecting high-dimensional data (3 features) into a lower-dimensional hidden layer (2 features).

This is a __Non-Square Matrix__. It is use in __Dimensionality Reduction__ to find the most important parts of a complex dataset.

    Formula: A(m*n)
    n is the input dimension
    m is the output dimension



