import math_code

#=============================================================================
# Vectors | Chapter 1: The Manual Neuron (Test)
#=============================================================================

print("Vectors | Chapter 1: The Manual Neuron")
weights = [0.7, -0.2, 0.5] # The vector
learning_rate = 0.01 # LeARNING rate (scalar)
print(f"Original Weight Vector: {weights}")
multiplied = math_code.scale_vector(weights, learning_rate)
math_code.add_vectors(weights, multiplied)


#=============================================================================
# Vectors | Chapter 2: Linear Combinations, Span, and Basis Vectors (Test)
#=============================================================================

print("\nVectors | Chapter 2: Linear Combinations, Span, and Basis Vectors")
v1 = [1, 0]
v2 = [0, 1]
v3 = [2, 3]

basis = [v1, v2]
if math_code.is_linear_combination(v3, basis):
    print(f"Vector {v3} is a linear combination of the basis vectors. (Redundant)")


#=============================================================================
# Vectors | Chapter 3: Linear Transformations and Matrices (Test)
#=============================================================================

print("\nVectors | Chapter 3: Linear Transformations and Matrices")

# Column 1 is where i hat landed, Column 2 is where j hat landed
matrix = [[1, -2], [3, 0]]
input_vector = [5, 2]

print(f"Transformed Vector: {math_code.transform_vector(matrix, input_vector)}")
