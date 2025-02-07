import torch  
import numpy as np  

def knapsack_specialized(xi, v, w, C):
    """
    Solves a specialized knapsack problem using a specialized method in a vectorized way

    Args:
        xi (torch.Tensor): xi variables.
        v (torch.Tensor): Quantization vector.
        w (torch.Tensor): Weight vector.
        C (int): Number of buckets of quantization.

    Returns:
        tuple: Optimal allocation (x_opt), optimal multipliers (lambda_opt), and objective values.
    """
    
    b_list = []
    b = 0

    # Compute breakpoint vector x_plus
    while True:
        delta_xi = (xi[b + 1:] - xi[b])
        delta_v = (v[b + 1:] - v[b])
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0

        if b != C - 1:
            b_list.append(int(b))

        if b + 1 > C - 1:
            break
    b_list.append(C - 1)
    x_plus = torch.zeros(C, dtype=torch.int32)
    b_tensor = torch.tensor(b_list, dtype=torch.int32)
    x_plus[b_tensor] = 1

    # Determine optimal allocation based on w
    w_idx = torch.searchsorted(v, w) 
    indices_breakpoints = torch.nonzero(x_plus == 1).squeeze()

    # Creation of masks for extreme cases
    mask_right = w > v[-1]
    mask_left = w < v[0]

    # Find indices using searchsorted
    search_idx = torch.searchsorted(indices_breakpoints, w_idx)

    # Ensure that the indices are valid
    search_idx = torch.clamp(search_idx, 1, len(indices_breakpoints) - 1)

    # Initialize idx_right and idx_left with the result of the search
    idx_right = indices_breakpoints[search_idx]
    idx_left = indices_breakpoints[search_idx - 1]

    # Correct the indices for extreme cases
    idx_right = torch.where(mask_right, indices_breakpoints[-1], idx_right)
    idx_left = torch.where(mask_right, indices_breakpoints[-1], idx_left)

    # Correct the indices for the case when w < v[0]
    idx_right = torch.where(mask_left, indices_breakpoints[0], idx_right)
    idx_left = torch.where(mask_left, indices_breakpoints[0], idx_left)

    # Compute convex combination for optimal solution
    x1, x2 = torch.zeros(2, len(w), C, dtype=torch.float32)

    x1[torch.arange(len(w)), idx_left] = 1
    x2[torch.arange(len(w)), idx_right] = 1

    numerator = w - torch.matmul(x2, v)
    denominator = torch.matmul((x1 - x2), v)
    theta = numerator / denominator

    mask_equal = (x1 == x2)
    theta_expanded = theta.unsqueeze(1)
    x_opt = torch.where(mask_equal, x1, x1 * theta_expanded + x2 * (1 - theta_expanded))

    # Compute optimal multipliers
    denominator = (v[idx_right] - v[idx_left])
    denominator_zero_mask = denominator == 0

    lambda_opt_nonzero = (xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = xi / v
    lambda_opt_zero_full[0] = 0
    lambda_opt_zero = lambda_opt_zero_full[idx_left]

    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # Compute objective function values
    objective_values = torch.matmul(x_opt, xi)

    return x_opt, lambda_opt, objective_values

# The following 2 functions serve for knapsack_specialized_single: the non-vectorized version of knapsack_specialized
def convex_combination(theta, i_left, i_right, C):
    """
    Creates a convex combination between two indices.

    Args:
        theta (float): Mixing coefficient.
        i_left (int): Left index.
        i_right (int): Right index.
        C (int): Dimension size.

    Returns:
        list: Convex combination vector.
    """
    combination = [0] * C
    combination[i_left] = theta
    combination[i_right] = 1 - theta
    return combination

def vector_one(i, C):
    """
    Creates a unit vector of size C with a 1 at index i.

    Args:
        i (int): Index where 1 is placed.
        C (int): Vector size.

    Returns:
        list: Unit vector.
    """
    vec = [0] * C
    vec[i] = 1
    return vec

# Non-vectorized version of knapsack_specialized. 
# It is utilized in the test phase and for comparison with other optimization techniques.
def knapsack_specialized_single(xi, v, w):
    """
    Optimized single-instance version of the knapsack algorithm.

    Args:
        xi (list or array): Ordered vector of size C.
        v (list or array): Ordered vector of size C.
        w (float): Scalar value belonging to v.

    Returns:
        tuple: Optimal solution vector, lambda value, optimal function value, iteration count.
    """
    C = len(xi)
    w_idx = v.index(w)  # Find the index corresponding to w in v

    i_left = 0
    b = 0  # Initialize variable b to 0
    iteration_count = 0

    while i_left <= C - 2:
        iteration_count += 1

        # Compute i_right by minimizing the ratio
        i_right = np.argmin([(xi[j] - xi[i_left]) / (v[j] - v[i_left]) if j > i_left else float('inf') for j in range(C)])

        # If b == 1, return direct allocation
        if b == 1:
            x_sol = vector_one(w_idx, C)
            lambda_ = (xi[i_right] - xi[i_left]) / (v[i_right] - v[i_left])
            optimal_value = np.dot(xi, x_sol)
            return x_sol, lambda_, optimal_value, iteration_count

        # Special case when w_idx == 0
        if w_idx == 0:
            x_sol = vector_one(0, C)
            lambda_ = (xi[i_right] - xi[i_left]) / (v[i_right] - v[i_left])
            optimal_value = np.dot(xi, x_sol)
            return x_sol, lambda_, optimal_value, iteration_count

        # Compute convex combination when w_idx is between i_left and i_right
        if i_left < w_idx < i_right:
            x1 = vector_one(i_left, C)
            x2 = vector_one(i_right, C)
            theta = (w - np.dot(v, x2)) / (np.dot(v, np.array(x1) - np.array(x2)))
            lambda_ = (xi[i_right] - xi[i_left]) / (v[i_right] - v[i_left])
            x_sol = convex_combination(theta, i_left, i_right, C)
            optimal_value = np.dot(xi, x_sol)
            return x_sol, lambda_, optimal_value, iteration_count

        # Special case when w_idx equals i_right
        if w_idx == i_right:
            if w_idx == C - 1:
                x_sol = vector_one(w_idx, C)
                lambda_ = (xi[i_right] - xi[i_left]) / (v[i_right] - v[i_left])
                optimal_value = np.dot(xi, x_sol)
                return x_sol, lambda_, optimal_value, iteration_count
            b = 1  # Set b to 1

        i_left = i_right  # Move to the next iteration
