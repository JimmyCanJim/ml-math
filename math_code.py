import random

def generate_big_batch(rows, cols):
    """Generates a large matrix of random data."""
    return [[random.uniform(0, 1000) for _ in range(cols)] for _ in range(rows)]

# Features: [Income, Age, Redundant_Income_Metric]
batch_X = generate_big_batch(100, 3)
# Manually make Column 3 redundant (Linear Dependence)
for row in batch_X:
    row[2] = row[0] * 1.5  

# 2x2 Weight Matrices for Layers 1 and 2
weights_layer1 = [[0.7, -0.2], [0.5, 0.1]] # 2x2
weights_layer2 = [[0.1, 0.9], [-0.4, 0.3]] # 2x2