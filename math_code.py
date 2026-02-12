import random

def generate_big_batch(rows, cols):
    """Generates a large matrix of random data."""
    return [[random.uniform(0, 1000) for _ in range(cols)] for _ in range(rows)]

#===================================================
# Module 1
#===================================================
def normalization(raw_data):
    num_rows = len(raw_data)
    num_cols = len(raw_data[0])

    col_maxes = [0.0] * num_cols
    for col in range(num_cols):
        current_max = 0.0
        for row in range(num_rows):
            val = abs(raw_data[row][col])
            if val > current_max:
                current_max = val
        
        col_maxes[col] = current_max

    normalized_matrix = []

    for row in range(num_rows):
        new_row = []
        for col in range(num_cols):
            if col_maxes[col] != 0:
                scaled_val = raw_data[row][col] * (1.0 / col_maxes[col])
            else:
                scaled_val = 0.0
            new_row.append(scaled_val)
        normalized_matrix.append(new_row)
    
    return normalized_matrix

#===================================================
# Module 2
#===================================================
def find_redundancy(matrix, col_a, col_b):
    num_rows = len(matrix)

    if matrix[0][col_a] != 0:
        c = matrix[0][col_b] / matrix[0][col_a]
    else:
        return False
    
    print(f"Candidate Constant Scalar (c): {c}")

    for i in range(1, num_rows):
        expected_val = matrix[i][col_a] * c
        actual_value = matrix[i][col_b]

        if abs(expected_val-actual_value) > 1e-9:
            print(f"Row {i} broke the pattern, not redundant.")
            return False
        
    print("Redundancy confirmed! Column B provides new information.")
    return True

#===================================================
# Module 3
#===================================================
def dense_layer():
    ...

if __name__ == "__main__":
    # Features: [Income, Age, Redundant_Income_Metric]
    batch_X = generate_big_batch(100, 3)
    # Manually make Column 3 redundant (Linear Dependence)
    for row in batch_X:
        row[2] = row[0] * 1.5  

    # 2x2 Weight Matrices for Layers 1 and 2
    weights_layer1 = [[0.7, -0.2], [0.5, 0.1]] # 2x2
    weights_layer2 = [[0.1, 0.9], [-0.4, 0.3]] # 2x2

    normalized_data = normalization(batch_X)
    is_redun = find_redundancy(batch_X, 0, 2)