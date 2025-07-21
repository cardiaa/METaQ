import torch   
import gc

def knapsack_specialized(xi, v, w, C, device):
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
    x_plus = torch.zeros(C, dtype=torch.int32, device=device)
    b_tensor = torch.tensor(b_list, dtype=torch.int32, device=device)
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
    x1, x2 = torch.zeros(2, len(w), C, dtype=torch.float32, device=device)

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

    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = -xi / v
    lambda_opt_zero_full[0] = 0
    lambda_opt_zero = lambda_opt_zero_full[idx_left]

    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # Compute objective function values
    objective_values = torch.matmul(x_opt, xi)

    del (xi, v, w, x_plus, b_list, b_tensor, indices_breakpoints, 
        w_idx, search_idx, idx_right, idx_left, x1, x2, numerator, 
        denominator, theta, mask_equal, theta_expanded)
    torch.cuda.empty_cache()    

    return x_opt, lambda_opt, objective_values

def knapsack_specialized_pruning(xi, v, w, C, device, delta):
    """
    Solves a specialized knapsack problem with pruning strategy, using vectorized operations.
    
    Args:
        xi (torch.Tensor): xi variables.
        v (torch.Tensor): Quantization vector.
        w (torch.Tensor): Weight vector.
        C (int): Number of quantization buckets.
        delta (float): Pruning threshold to adjust xi.
        device (torch.device): Target device for computation.

    Returns:
        tuple: Optimal allocation (x), optimal multipliers (lambda_opt), and objective values.
    """
    print("inside function", flush=True) # Debugging line
    #print("Begin knaspasck_specialized_pruning ...") # Debugging line
    xi = xi.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)
    w = w.to(dtype=torch.float32, device=device)

    xi = xi - delta
    M = w.shape[0]
    
    # === Step 1: Compute x_plus ===
    b_list = []
    b = 0
    #print("Computing x_plus...") # Questo l'ho messo solo per vedere se il calcolo di x_plus era il collo di bottiglia, e non lo è.
    while True:
        delta_xi = xi[b + 1:] - xi[b]
        delta_v = v[b + 1:] - v[b]
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0
        if b != C - 1:
            b_list.append(int(b))
        if b + 1 > C - 1:
            break
    b_list.append(C - 1)
    x_plus = torch.zeros(C, dtype=torch.int32)
    x_plus[torch.tensor(b_list)] = 1
    x_plus = x_plus.to(device)
    #print("x_plus calculated")
    """
    x_plus = torch.zeros(C, dtype=torch.int32)
    x_plus[0] = 1
    x_plus[-1] = 1
    x_plus = x_plus.to(device)
    """
    # === Step 2: Precompute ===
    ratio = xi / v
    neg_indices = torch.where(v < 0)[0]
    pos_indices = torch.where(v >= 0)[0]
    neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
    pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
    b_vector = torch.cat([neg_sorted, pos_sorted], dim=0)
    b_vector = b_vector.to(device)
    #print(w.device, v.device, xi.device, b_vector.device) # ... debug ...

    # === Step 3: Masks ===
    mask_small = w < v[0]
    mask_large = w > v[-1]
    mask_mid = (~mask_small) & (~mask_large)
    mask_edge = mask_small | mask_large

    # === Step 4: Initialize outputs ===
    x = torch.zeros(M, C, device=device)
    lambda_opt = torch.zeros(M, device=device)

    # === Step 5: Edge cases ===
    #print("Processing edge cases...") # Debugging line
    if mask_edge.any():
        w_edge = w[mask_edge]
        x_edge = torch.zeros((w_edge.shape[0], C), device=device, dtype=torch.float32)

        # Divisioni per v[0] e v[-1]
        w_div_v0 = w_edge / v[0]
        w_div_v_last = w_edge / v[-1]

        # Masks per sotto-casi
        edge_small = w_edge < v[0]
        edge_large = w_edge > v[-1]

        # Per w < v[0]
        mask_cond_small = (w_div_v0 >= 0) & (w_div_v0 <= 1) & edge_small
        mask_else_small = edge_small & (~mask_cond_small)

        if mask_cond_small.any():
            # Calcola (w / v) * xi per ogni w, in modo broadcasting: w_edge_i / v_j * xi_j
            # Qui serve w_edge_i / v_j per tutti i j, e moltiplichiamo per xi_j
            # xi e v sono vettori di dimensione C
            # Otteniamo matrice M_mid x C

            w_small = w_edge[mask_cond_small].unsqueeze(1)  # shape (N_s, 1)
            # broadcasting divisione
            div_mat = w_small / v.unsqueeze(0)  # (N_s, C)
            val_mat = div_mat * xi.unsqueeze(0)  # (N_s, C)

            # Trova argmin per ogni riga
            i_min = torch.argmin(val_mat, dim=1)
            vals_min = div_mat[torch.arange(i_min.shape[0]), i_min]

            # Assegna a x_edge
            x_edge[mask_cond_small, :] = 0
            x_edge[mask_cond_small, i_min] = vals_min

        if mask_else_small.any():
            x_edge[mask_else_small, 0] = 1.0

        # Per w > v[-1]
        mask_cond_large = (w_div_v_last >= 0) & (w_div_v_last <= 1) & edge_large
        mask_else_large = edge_large & (~mask_cond_large)

        if mask_cond_large.any():
            w_large = w_edge[mask_cond_large].unsqueeze(1)
            div_mat = w_large / v.unsqueeze(0)
            val_mat = div_mat * xi.unsqueeze(0)
            i_min = torch.argmin(val_mat, dim=1)
            vals_min = div_mat[torch.arange(i_min.shape[0]), i_min]
            x_edge[mask_cond_large, :] = 0
            x_edge[mask_cond_large, i_min] = vals_min

        if mask_else_large.any():
            x_edge[mask_else_large, -1] = 1.0

        x[mask_edge] = x_edge
    #print("end Processing edge cases...") # Debugging line
    #print("Processing mid cases A ...") # Debugging line
    # === Step 6: Intermediate Case ===
    if mask_mid.any():
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # First method
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector] == 1).unsqueeze(0)
        valid_i0 = torch.where(valid, torch.arange(C, device=device)[None, :], torch.tensor(float('inf'), device=device))
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]
        x1_sol = torch.zeros(M_mid, C, device=device)
        theta1 = w_mid / v_i0
        x1_sol[torch.arange(M_mid), i0] = theta1
        obj1 = x1_sol @ xi
        obj1[theta1 < 0] = torch.tensor(float('inf'), device=device)

        # Second method
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0]
        one_indices = one_indices.to(device=device, dtype=torch.long)
        i_right = torch.searchsorted(v[one_indices], w_mid, right=False)
        i_right = i_right.clamp(min=1, max=one_indices.shape[0] - 1)
        idx_right_mid = one_indices[i_right]
        idx_left_mid = one_indices[i_right - 1]
        v_left = v[idx_left_mid]
        v_right = v[idx_right_mid]
        theta2 = (w_mid - v_right) / (v_left - v_right + 1e-8)

        x2_sol = torch.zeros(M_mid, C, device=device)
        x2_sol[torch.arange(M_mid), idx_left_mid] = theta2
        x2_sol[torch.arange(M_mid), idx_right_mid] = 1 - theta2
        obj2 = x2_sol @ xi

        # Choose better
        better_first = obj1 < obj2
        final_x = torch.where(better_first.unsqueeze(1), x1_sol, x2_sol)
        x[mask_mid] = final_x
    #print("end Processing mid cases A ...") # Debugging line
    # === Step 7: Compute idx_left and idx_right globally ===
    one_indices = torch.nonzero(x_plus, as_tuple=True)[0]

    idx_left = torch.zeros_like(w, dtype=torch.long)
    idx_right = torch.zeros_like(w, dtype=torch.long)

    # Mid case
    #print("Processing mid cases B ...") # Debugging line
    if mask_mid.any():
        i_right_mid = torch.searchsorted(v[one_indices], w[mask_mid], right=False)
        i_right_mid = i_right_mid.clamp(min=1, max=one_indices.shape[0] - 1)
        idx_right_mid = one_indices[i_right_mid]
        idx_left_mid = one_indices[i_right_mid - 1]

        i0_full = torch.zeros_like(w, dtype=torch.long)
        better_first_full = torch.zeros_like(w, dtype=torch.bool)
        i0_full[mask_mid] = i0
        better_first_full[mask_mid] = better_first

        idx_left[mask_mid] = torch.where(better_first, i0, idx_left_mid)
        idx_right[mask_mid] = torch.where(better_first, i0, idx_right_mid)
    #print("end Processing mid cases B ...") # Debugging line
    # Edge case
    #print("Calculating indices ...") # Debugging line
    if mask_edge.any():
        x_edge_masked = x[mask_edge]  # (N_edge, C)
        idx_edge = torch.nonzero(x_edge_masked, as_tuple=True)[1]  # colonna (indice lungo C)
        idx_left[mask_edge] = idx_edge
        idx_right[mask_edge] = idx_edge
    #print("end Calculating indices ...") # Debugging line
    # === Step 8: Compute lambda_opt ===
    #print("Start part A ...") # Debugging line
    denominator = v[idx_right] - v[idx_left]
    denominator_zero_mask = denominator == 0

    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = -(xi + delta) / v
    lambda_opt_zero = lambda_opt_zero_full[idx_left]
    #print("End part A ...") # Debugging line
    #print("Start part B ...") # Debugging line
    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)
    #print("End part B ...") # Debugging line
    # === Step 9: Objective ===
    objective_values = delta + x @ xi

    #print("=== Device Report ===")
    #for name, obj in locals().items():
    #    if isinstance(obj, torch.Tensor):
    #        print(f"{name:25s} -> {obj.device}")
    #print("=====================")

    # Cleanup: delete intermediate tensors
    #print("Start part A ...") # Debugging line
    for var in [
        'x_edge', 'x1_sol', 'x2_sol', 'val_mat', 'div_mat', 'final_x', 
        'ratio', 'neg_indices', 'pos_indices', 'neg_sorted', 'pos_sorted', 
        'b_vector', 'one_indices', 'idx_left', 'idx_right', 'idx_left_mid', 'idx_right_mid',
        'theta1', 'theta2', 'obj1', 'obj2', 'i0', 'i0_pos', 'valid', 'valid_i0',
        'lambda_opt_nonzero', 'lambda_opt_zero', 'lambda_opt_zero_full',
        'denominator', 'denominator_zero_mask',
        'edge_small', 'edge_large', 'mask_cond_small', 'mask_else_small',
        'mask_cond_large', 'mask_else_large', 'w_edge', 'w_mid',
        'mask_edge', 'mask_mid', 'mask_small', 'mask_large'
    ]:
        if var in locals():
            del locals()[var]
    #print("End part A ...") # Debugging line
    #print("Start part B ...") # Debugging line
    # Garbage collection & CUDA cache
    gc.collect()
    torch.cuda.empty_cache()  
    #print("End part B ...") # Debugging line
    #print("End knaspasck_specialized_pruning ...") # Debugging line
    return x, lambda_opt, objective_values

def knapsack_specialized_histo(xi, v, w, C, device):
    """
    Solves the specialized knapsack problem in the vectorized way to construct the histogram in the complexity analysis

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
    iterations = 0
    # Compute breakpoint vector x_plus
    while True:
        iterations += 1
        delta_xi = (xi[b + 1:] - xi[b])
        delta_v = (v[b + 1:] - v[b])
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0

        if b != C - 1:
            b_list.append(int(b))

        if b + 1 > C - 1:
            break
    b_list.append(C - 1)
    x_plus = torch.zeros(C, dtype=torch.int32, device=device)
    b_tensor = torch.tensor(b_list, dtype=torch.int32, device=device)
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
    x1, x2 = torch.zeros(2, len(w), C, dtype=torch.float32, device=device)

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

    return x_opt, lambda_opt, objective_values, iterations

