import torch
import gc
import math

def _make_effective_pruning_xi(xi, C, device, delta):
    """
    Builds the effective cost vector for the sparse-aware pruning knapsack.

    If xi has length C + 1:
        xi[0]  = multiplier for the zero/pruning symbol
        xi[1:] = multipliers for the non-zero buckets

    The per-weight sparse-aware knapsack, after eliminating z_i, has effective
    non-zero bucket costs:

        xi_b - xi_zero - delta

    If xi has length C, this falls back to the previous behavior:

        xi_b - delta
    """
    xi = xi.to(dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)

    if xi.numel() == C + 1:
        xi_zero = xi[0]
        xi_buckets = xi[1:]
        xi_effective = xi_buckets - xi_zero - delta_t
        objective_constant = xi_zero + delta_t
    else:
        xi_effective = xi - delta_t
        objective_constant = delta_t

    return xi_effective, objective_constant

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
    #print("inside function", flush=True) # Debugging line
    #print("Begin knaspasck_specialized_pruning ...", flush=True) # Debugging line
    #print("debug 1", flush=True) # Debugging line
    v = v.to(dtype=torch.float32, device=device)
    w = w.to(dtype=torch.float32, device=device)

    xi, objective_constant = _make_effective_pruning_xi(
        xi,
        C,
        device,
        delta,
    )

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
    #print("debug 2", flush=True) # Debugging line
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
    #print("debug 3", flush=True) # Debugging line
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
    #print("debug 4", flush=True) # Debugging line
    #print("end Processing edge cases...") # Debugging line
    #print("Processing mid cases A ...") # Debugging line
    # === Step 6: Intermediate Case ===
    if mask_mid.any():
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # First method
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector] == 1).unsqueeze(0)
        valid_i0 = torch.where(
            valid,
            torch.arange(C, device=device)[None, :],
            torch.tensor(float('inf'), device=device)
        )
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]

        theta1 = w_mid / v_i0
        x1_sol = torch.zeros(M_mid, C, device=device)
        x1_sol.scatter_(1, i0.unsqueeze(1), theta1.unsqueeze(1))
        obj1 = x1_sol @ xi
        obj1[theta1 < 0] = torch.tensor(float('inf'), device=device)

        # Second method
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0].to(device=device, dtype=torch.long)
        i_right = torch.searchsorted(v[one_indices], w_mid, right=False)
        i_right = i_right.clamp(min=1, max=one_indices.shape[0] - 1)

        idx_right_mid = one_indices[i_right]
        idx_left_mid = one_indices[i_right - 1]
        v_left = v[idx_left_mid]
        v_right = v[idx_right_mid]
        theta2 = (w_mid - v_right) / (v_left - v_right + 1e-8)

        x2_sol = torch.zeros(M_mid, C, device=device)
        x2_sol.scatter_(1, idx_left_mid.unsqueeze(1), theta2.unsqueeze(1))
        x2_sol.scatter_(1, idx_right_mid.unsqueeze(1), (1 - theta2).unsqueeze(1))
        obj2 = x2_sol @ xi

        # Choose better
        better_first = obj1 < obj2
        final_x = torch.where(better_first.unsqueeze(1), x1_sol, x2_sol)
        x[mask_mid] = final_x

    #print("end Processing mid cases A ...") # Debugging line
    # === Step 7: Compute idx_left and idx_right globally ===
    #print("debug 5", flush=True) # Debugging line
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
    #print("debug 6", flush=True) # Debugging line
    #print("end Processing mid cases B ...") # Debugging line
    # Edge case
    #print("Calculating indices ...") # Debugging line
    if mask_edge.any():
        x_edge_masked = x[mask_edge]  # (N_edge, C)
        idx_edge = torch.nonzero(x_edge_masked, as_tuple=True)[1]  # colonna (indice lungo C)
        idx_left[mask_edge] = idx_edge
        idx_right[mask_edge] = idx_edge
    #print("debug 7", flush=True) # Debugging line
    #print("end Calculating indices ...") # Debugging line
    # === Step 8: Compute lambda_opt ===
    #print("Start part A ...") # Debugging line
    denominator = v[idx_right] - v[idx_left]
    denominator_zero_mask = denominator == 0

    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = - xi / v
    lambda_opt_zero = lambda_opt_zero_full[idx_left]
    #print("End part A ...") # Debugging line
    #print("Start part B ...") # Debugging line
    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)
    #print("End part B ...") # Debugging line
    # === Step 9: Objective ===
    objective_values = objective_constant + x @ xi

    #print("=== Device Report ===")
    #for name, obj in locals().items():
    #    if isinstance(obj, torch.Tensor):
    #        print(f"{name:25s} -> {obj.device}")
    #print("=====================")

    # Cleanup: delete intermediate tensors
    #print("Start part A ...") # Debugging line
    #print("debug 8", flush=True) # Debugging line
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
    #print("End knaspasck_specialized_pruning ...", flush=True) # Debugging line
    return x, lambda_opt, objective_values

def knapsack_specialized_pruning_sparse(xi, v, w, C, device, delta):
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
    v = v.to(dtype=torch.float32, device=device)
    w = w.to(dtype=torch.float32, device=device)

    xi, objective_constant = _make_effective_pruning_xi(
        xi,
        C,
        device,
        delta,
    )

    M = w.shape[0]
    
    # === Step 1: Compute x_plus ===
    b_list = []
    b = 0
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

    # === Step 2: Precompute ===
    ratio = xi / v
    neg_indices = torch.where(v < 0)[0]
    pos_indices = torch.where(v >= 0)[0]
    neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
    pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
    b_vector = torch.cat([neg_sorted, pos_sorted], dim=0)
    b_vector = b_vector.to(device)

    # === Step 3: Masks ===
    mask_small = w < v[0]
    mask_large = w > v[-1]
    mask_mid = (~mask_small) & (~mask_large)
    mask_edge = mask_small | mask_large

    # === Step 4: Initialize outputs ===
    x = torch.zeros(M, C, device=device)
    lambda_opt = torch.zeros(M, device=device)

    # === Step 5: Edge cases ===
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
            w_small = w_edge[mask_cond_small].unsqueeze(1)
            div_mat = w_small / v.unsqueeze(0)
            val_mat = div_mat * xi.unsqueeze(0)
            i_min = torch.argmin(val_mat, dim=1)
            vals_min = div_mat[torch.arange(i_min.shape[0]), i_min]
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

    # === Step 6: Intermediate Case ===
    if mask_mid.any():
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # First method
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector] == 1).unsqueeze(0)
        valid_i0 = torch.where(
            valid,
            torch.arange(C, device=device)[None, :],
            torch.tensor(float('inf'), device=device)
        )
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]

        theta1 = w_mid / v_i0
        x1_sol = torch.zeros(M_mid, C, device=device)
        x1_sol.scatter_(1, i0.unsqueeze(1), theta1.unsqueeze(1))
        obj1 = x1_sol @ xi
        obj1[theta1 < 0] = torch.tensor(float('inf'), device=device)

        # Second method
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0].to(device=device, dtype=torch.long)
        i_right = torch.searchsorted(v[one_indices], w_mid, right=False)
        i_right = i_right.clamp(min=1, max=one_indices.shape[0] - 1)
        idx_right_mid = one_indices[i_right]
        idx_left_mid = one_indices[i_right - 1]
        v_left = v[idx_left_mid]
        v_right = v[idx_right_mid]
        theta2 = (w_mid - v_right) / (v_left - v_right + 1e-8)

        x2_sol = torch.zeros(M_mid, C, device=device)
        x2_sol.scatter_(1, idx_left_mid.unsqueeze(1), theta2.unsqueeze(1))
        x2_sol.scatter_(1, idx_right_mid.unsqueeze(1), (1 - theta2).unsqueeze(1))
        obj2 = x2_sol @ xi

        # Choose better
        better_first = obj1 < obj2
        final_x = torch.where(better_first.unsqueeze(1), x1_sol, x2_sol)
        x[mask_mid] = final_x

    # === Step 7: Compute idx_left and idx_right globally ===
    one_indices = torch.nonzero(x_plus, as_tuple=True)[0]
    idx_left = torch.zeros_like(w, dtype=torch.long)
    idx_right = torch.zeros_like(w, dtype=torch.long)
    
    if mask_mid.any():
        i_right_mid = torch.searchsorted(v[one_indices], w[mask_mid], right=False)
        i_right_mid = i_right_mid.clamp(min=1, max=one_indices.shape[0] - 1)
        idx_right_mid = one_indices[i_right_mid]
        idx_left_mid = one_indices[i_right_mid - 1]
        idx_left[mask_mid] = torch.where(better_first, i0, idx_left_mid)
        idx_right[mask_mid] = torch.where(better_first, i0, idx_right_mid)

    if mask_edge.any():
        x_edge_masked = x[mask_edge]
        idx_edge = torch.nonzero(x_edge_masked, as_tuple=True)[1]
        idx_left[mask_edge] = idx_edge
        idx_right[mask_edge] = idx_edge

    # === Step 8: Compute lambda_opt ===
    denominator = v[idx_right] - v[idx_left]
    denominator_zero_mask = denominator == 0
    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = - xi / v
    lambda_opt_zero = lambda_opt_zero_full[idx_left]
    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # === Step 9: Objective ===
    objective_values = objective_constant + x @ xi

    # === Cleanup (snella, stile funzione sparsa) ===
    del ratio, neg_indices, pos_indices, neg_sorted, pos_sorted, b_vector

    return x, lambda_opt, objective_values

def knapsack_perspective_leonardo(xi_buckets, v, w, C, device, T1, T2, T3):
    """
    General (T2 != 0) per-weight perspective subproblem solver.

    Solves, for each weight w:
        min_x  sum_b (T2*xi_b) x_b  +  T1 w^2/y  +  T3 y,
        s.t.  sum_b v_b x_b = w,  y = sum_b x_b,  y <= 1,  x >= 0.

    via the reduction K(y) = y*K(1)[w/y] and the 1-D convex minimisation of
    G(y) = K(y) + T1 w^2/y + T3 y over y in [ymin, 1], done as a segment-scan on
    the lower convex envelope of the (v_b, T2*xi_b) points.  Candidates on each
    envelope segment j (slope s_j, offset beta_j): the stationary point
    y_j = |w| sqrt(T1/(s_j+T3)), the two breakpoints (kinks) w/V[j], w/V[j+1],
    and the global endpoints ymin, 1.  G convex => the min over accepted
    candidates is global.  The offline verification against cvxpy is in
    CheckCorrectnessPerspectiveAlgorithm.ipynb / scratchpad/verify_persp_general.py.

    Returns:
        x_placeholder: (M, 4) = [idx_left, idx_right, x_left, x_right]
                       (bucket masses; x_left+x_right = y*, and z = 1 - y*)
        beta_star:     (M,)  full gradient dphi/dw = beta_old(w/y*) + 2 T1 w/y*
        y_star:        (M,)
    """
    v = v.to(dtype=torch.float32, device=device)
    w = w.to(dtype=torch.float32, device=device)
    # The x-subproblem cost is the RAW dual xi_b (verified offline).  T2 does NOT
    # multiply the bucket costs here; it enters only the entropy-dual relation
    # c_b* = exp(log2*xi_b/T2 - 1) in the FISTA update.  (T2 is accepted for API
    # symmetry but unused in this solver.)
    xi_eff = xi_buckets.to(dtype=torch.float32, device=device)
    M = w.shape[0]

    # --- lower convex envelope (same breakpoint logic as the other knapsacks) ---
    b_list = []
    b = 0
    while True:
        delta_xi = xi_eff[b + 1:] - xi_eff[b]
        delta_v = v[b + 1:] - v[b]
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0
        if b != C - 1:
            b_list.append(int(b))
        if b + 1 > C - 1:
            break
    b_list.append(C - 1)
    x_plus = torch.zeros(C, dtype=torch.int32, device=device)
    x_plus[0] = 1
    x_plus[torch.tensor(b_list, device=device, dtype=torch.long)] = 1
    one = torch.nonzero(x_plus, as_tuple=True)[0]          # envelope vertex indices, sorted by v
    V = v[one]
    Xi = xi_eff[one]
    P = V.numel()
    dV = V[1:] - V[:-1]
    beta_seg = (Xi[1:] - Xi[:-1]) / dV                     # (P-1,) slope of K(1) on segment j
    s_seg = (Xi[:-1] * V[1:] - Xi[1:] * V[:-1]) / dV        # (P-1,) intercept s_j

    aw = w.abs()
    pos_max = V[-1].clamp_min(1e-12)
    neg_absmax = V[0].abs().clamp_min(1e-12)
    side = torch.where(w >= 0, pos_max, neg_absmax)
    ymin = (aw / side).clamp_(max=1.0)
    ones = torch.ones_like(w)
    eps = 1e-6

    best_G = torch.full((M,), float('inf'), device=device, dtype=torch.float32)
    best_y = ones.clone()

    def _consider(cand, sj, bj, Vl, Vr, allow):
        cand = torch.minimum(torch.maximum(cand, ymin), ones)
        what = w / cand.clamp_min(1e-30)
        in_seg = allow & (what >= Vl - eps) & (what <= Vr + eps) & (aw > 0)
        G = bj * w + sj * cand + T1 * w * w / cand.clamp_min(1e-30) + T3 * cand
        G = torch.where(in_seg, G, best_G.new_full((M,), float('inf')))
        upd = G < best_G
        best_G[upd] = G[upd]
        best_y[upd] = cand[upd]

    all_true = torch.ones_like(w, dtype=torch.bool)
    for j in range(P - 1):
        Vl = V[j]; Vr = V[j + 1]; sj = s_seg[j]; bj = beta_seg[j]
        d = float(s_seg[j]) + T3                            # s_j + T3 (scalar)
        if d > 0:
            ycand = aw * math.sqrt(T1 / d)                  # stationary point on segment j
            _consider(ycand, sj, bj, Vl, Vr, all_true)
        _consider(ymin.clone(), sj, bj, Vl, Vr, all_true)
        _consider(ones.clone(), sj, bj, Vl, Vr, all_true)
        _consider(w / Vl, sj, bj, Vl, Vr, all_true)         # breakpoint (kink) at wh=Vl
        _consider(w / Vr, sj, bj, Vl, Vr, all_true)         # breakpoint (kink) at wh=Vr

    y_star = best_y.clamp_min(1e-12)
    what = w / y_star
    seg = torch.searchsorted(V, what).clamp_(1, P - 1) - 1  # bracketing segment of w/y*
    beta_old = beta_seg[seg]
    # bucket masses: at wh in [V[seg],V[seg+1]], K(1) uses buckets one[seg], one[seg+1]
    Vl = V[seg]; Vr = V[seg + 1]
    theta_u = ((Vr - what) / (Vr - Vl)).clamp_(0.0, 1.0)    # mass fraction on the left bucket
    idx_left = one[seg]
    idx_right = one[seg + 1]
    x_left = y_star * theta_u
    x_right = y_star * (1.0 - theta_u)
    beta_star = beta_old + 2.0 * T1 * w / y_star

    # weights that are exactly zero -> fully pruned, no gradient
    zero_w = aw <= 0
    x_left = torch.where(zero_w, torch.zeros_like(x_left), x_left)
    x_right = torch.where(zero_w, torch.zeros_like(x_right), x_right)
    beta_star = torch.where(zero_w, torch.zeros_like(beta_star), beta_star)

    x_placeholder = torch.stack(
        [idx_left.to(torch.int32), idx_right.to(torch.int32), x_left, x_right],
        dim=1,
    )  # (M, 4)
    return x_placeholder, beta_star, y_star


def prox_perspective_leonardo(xi_buckets, v, u, C, device, T1, T3, gamma):
    """
    test_135: PROXIMAL per-weight subproblem.  Same model, different algorithm.

    Instead of returning a subgradient to be summed into the loss gradient, this
    solves the proximal operator of phi directly, per weight:

        prox_{gamma*phi}(u) = argmin_z { phi(z) + (1/(2 gamma)) (z-u)^2 }

    i.e., per weight,

        min_{z,x,y}  sum_b xi_b x_b + T1 z^2/y + T3 y + (1/(2 gamma)) (z-u)^2
        s.t.  sum_b v_b x_b = z,  sum_b x_b = y,  y <= 1,  x >= 0.

    The weight itself is now a VARIABLE (z) rather than a fixed target, so the
    solver returns the new weight z* directly.  The training loop then takes a
    plain gradient step on the loss alone and applies this operator to the
    weights: the entropy displacement no longer competes with the loss gradient
    and is no longer throttled by the learning rate -- which test_134 identified
    as the actual bottleneck (fixing the dual bought only +11%).

    Derivation (see the LaTeX "Fase 9").  On envelope segment j, where
    K(1)[zh] = beta_j*zh + s_j, the objective is
        G(z,y) = beta_j z + s_j y + T1 z^2/y + T3 y + (1/(2 gamma))(z-u)^2,
    and the two stationarity conditions give
        dG/dy = 0  =>  y = |z| sqrt(T1/(s_j+T3))          (same as the non-prox case)
        dG/dz = 0  =>  z = u - gamma*beta_j - 2 gamma sign(z) sqrt(T1 (s_j+T3)),
    the latter being a shifted soft-threshold, so z = 0 is a genuine outcome:
    pruning falls out of the operator instead of needing the separate magnitude
    rule.  Candidates per segment are that interior point, the top edge y = 1
    (closed form z = (u - gamma beta_j)/(1 + 2 gamma T1)), and the two cone edges
    z/y = V_j, V_{j+1}; plus the global "prune" candidate z = y = 0.  G is convex,
    so the min over accepted candidates is global.

    Verified offline against cvxpy (scratchpad/verify_prox.py, verify_prox_edge.py):
    over ~750 random instances spanning gamma in [1e-4, 1e2], u from ~0 to the
    distribution tail and T3 up to 1e-1, the closed form was never worse than the
    exact optimum; max |z* - z*_cvxpy| ~ 1e-7.

    Returns:
        x_placeholder: (M, 4) = [idx_left, idx_right, x_left, x_right]
        z_star:        (M,)  the new weights
        y_star:        (M,)
    """
    v = v.to(dtype=torch.float32, device=device)
    u = u.to(dtype=torch.float32, device=device)
    xi_eff = xi_buckets.to(dtype=torch.float32, device=device)
    M = u.shape[0]

    # --- lower convex envelope (identical to knapsack_perspective_leonardo) ---
    b_list = []
    b = 0
    while True:
        delta_xi = xi_eff[b + 1:] - xi_eff[b]
        delta_v = v[b + 1:] - v[b]
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0
        if b != C - 1:
            b_list.append(int(b))
        if b + 1 > C - 1:
            break
    b_list.append(C - 1)
    x_plus = torch.zeros(C, dtype=torch.int32, device=device)
    x_plus[0] = 1
    x_plus[torch.tensor(b_list, device=device, dtype=torch.long)] = 1
    one = torch.nonzero(x_plus, as_tuple=True)[0]
    V = v[one]
    Xi = xi_eff[one]
    P = V.numel()
    dV = V[1:] - V[:-1]
    beta_seg = (Xi[1:] - Xi[:-1]) / dV
    s_seg = (Xi[:-1] * V[1:] - Xi[1:] * V[:-1]) / dV

    eps = 1e-6
    zeros = torch.zeros_like(u)
    # candidate 0: fully pruned (z = y = 0), cost (1/(2 gamma)) u^2
    best_G = (0.5 / gamma) * u * u
    best_z = zeros.clone()
    best_y = zeros.clone()

    def _consider(zc, yc, bj, sj, Vl, Vr, allow):
        """Accept (zc,yc) if it lies in this segment's cone and improves G."""
        yc = yc.clamp(min=0.0, max=1.0)
        pos = yc > 1e-12
        zh = torch.where(pos, zc / yc.clamp_min(1e-30), torch.zeros_like(zc))
        in_seg = allow & pos & (zh >= Vl - eps) & (zh <= Vr + eps)
        G = (bj * zc + sj * yc + T1 * zc * zc / yc.clamp_min(1e-30)
             + T3 * yc + (0.5 / gamma) * (zc - u) ** 2)
        G = torch.where(in_seg, G, torch.full_like(G, float('inf')))
        upd = G < best_G
        best_G[upd] = G[upd]
        best_z[upd] = zc[upd]
        best_y[upd] = yc[upd]

    all_true = torch.ones_like(u, dtype=torch.bool)
    for j in range(P - 1):
        Vl = V[j]; Vr = V[j + 1]; bj = beta_seg[j]; sj = s_seg[j]
        d = float(s_seg[j]) + T3

        # (1) interior in y: shifted soft-threshold, with sign consistency
        if d > 0.0:
            A = gamma * bj
            B = 2.0 * gamma * math.sqrt(T1 * d)
            root = math.sqrt(T1 / d)
            z_pos = u - A - B
            _consider(z_pos, z_pos.abs() * root, bj, sj, Vl, Vr, z_pos > 0)
            z_neg = u - A + B
            _consider(z_neg, z_neg.abs() * root, bj, sj, Vl, Vr, z_neg < 0)

        # (2) top edge y = 1
        z_top = (u - gamma * bj) / (1.0 + 2.0 * gamma * T1)
        _consider(z_top, torch.ones_like(u), bj, sj, Vl, Vr, all_true)

        # (3)/(4) cone edges z/y = Vl and z/y = Vr
        for Vx in (Vl, Vr):
            Vx_f = float(Vx)
            if abs(Vx_f) < 1e-14:
                continue
            Xix = bj * Vx + sj                      # K(1)[Vx] at that vertex
            y_edge = (u - gamma * (Xix + T3 + T1 * Vx * Vx) / Vx) / Vx
            y_edge = y_edge.clamp(min=0.0, max=1.0)
            _consider(Vx * y_edge, y_edge, bj, sj, Vl, Vr, all_true)

    z_star = best_z
    y_star = best_y

    # bucket masses of the accepted solution, for the dual's count accumulation
    safe_y = y_star.clamp_min(1e-30)
    what = torch.where(y_star > 1e-12, z_star / safe_y, torch.zeros_like(z_star))
    seg = torch.searchsorted(V, what).clamp_(1, P - 1) - 1
    Vl_s = V[seg]; Vr_s = V[seg + 1]
    theta_u = ((Vr_s - what) / (Vr_s - Vl_s)).clamp_(0.0, 1.0)
    idx_left = one[seg]
    idx_right = one[seg + 1]
    x_left = y_star * theta_u
    x_right = y_star * (1.0 - theta_u)
    pruned = y_star <= 1e-12
    x_left = torch.where(pruned, torch.zeros_like(x_left), x_left)
    x_right = torch.where(pruned, torch.zeros_like(x_right), x_right)

    x_placeholder = torch.stack(
        [idx_left.to(torch.int32), idx_right.to(torch.int32), x_left, x_right],
        dim=1,
    )
    return x_placeholder, z_star, y_star


def knapsack_specialized_pruning_sparse_leonardo(xi, v, w, C, device, delta):
    """
    Memory-light version of knapsack_specialized_pruning with same logic as the dense version,
    but without materializing the dense x of shape (M, C).

    Returns:
        x_placeholder: (M, 3) with [idx_left, idx_right, theta] (int32, int32, float32)
        lambda_opt:    (M,)
        objective_values: (M,)
    """

    # === Step 0: Cast + move ===
    v = v.to(dtype=torch.float32, device=device)
    w = w.to(dtype=torch.float32, device=device)

    # Sparse-aware pruning costs.
    # If xi has length C + 1, xi[0] is the zero/pruning multiplier and xi[1:]
    # are the non-zero bucket multipliers.  The effective non-zero costs are:
    #
    #     xi_b - xi_zero - delta
    #
    # If xi has length C, this falls back to the previous behavior xi_b - delta.
    xi, objective_constant = _make_effective_pruning_xi(
        xi,
        C,
        device,
        delta,
    )

    M = w.shape[0]

    # === Step 1: Compute x_plus (same as dense) ===
    b_list = []
    b = 0
    while True:
        delta_xi = xi[b + 1:] - xi[b]
        delta_v = v[b + 1:] - v[b]
        b = torch.argmin(delta_xi / delta_v) + 1 + b_list[-1] if b_list else 0
        if b != C - 1:
            b_list.append(int(b))
        if b + 1 > C - 1:
            break
    b_list.append(C - 1)

    x_plus = torch.zeros(C, dtype=torch.int32, device=device)
    x_plus[torch.tensor(b_list, device=device, dtype=torch.long)] = 1

    # === Step 2: Precompute (same as dense; kept for identical behavior) ===
    ratio = xi / v
    neg_indices = torch.where(v < 0)[0]
    pos_indices = torch.where(v >= 0)[0]
    neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
    pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
    b_vector = torch.cat([neg_sorted, pos_sorted], dim=0).to(device)

    # === Step 3: Masks (same as dense) ===
    mask_small = w < v[0]
    mask_large = w > v[-1]
    mask_mid = (~mask_small) & (~mask_large)
    mask_edge = mask_small | mask_large

    # === Step 4: Outputs in 2-sparse form ===
    idx_left = torch.zeros(M, dtype=torch.long, device=device)
    idx_right = torch.zeros(M, dtype=torch.long, device=device)
    theta = torch.zeros(M, dtype=torch.float32, device=device)

    # === Step 5: Edge cases (IDENTICAL BRANCHING to dense) ===
    if mask_edge.any():
        edge_idx = torch.nonzero(mask_edge, as_tuple=True)[0]
        w_edge = w[mask_edge]

        # Precompute ratios used in dense
        w_div_v0 = w_edge / v[0]
        w_div_v_last = w_edge / v[-1]

        edge_small = w_edge < v[0]
        edge_large = w_edge > v[-1]

        # ---- w < v[0] ----
        mask_cond_small = (w_div_v0 >= 0) & (w_div_v0 <= 1) & edge_small
        mask_else_small = edge_small & (~mask_cond_small)

        if mask_cond_small.any():
            inds = edge_idx[mask_cond_small]
            w_small = w_edge[mask_cond_small].unsqueeze(1)          # (k,1)
            div_mat = w_small / v.unsqueeze(0)                       # (k,C)
            val_mat = div_mat * xi.unsqueeze(0)                      # (k,C)
            i_min = torch.argmin(val_mat, dim=1)                     # (k,)
            vals_min = div_mat[torch.arange(i_min.shape[0], device=device), i_min]

            idx_left[inds] = i_min
            idx_right[inds] = i_min
            theta[inds] = vals_min

        if mask_else_small.any():
            inds = edge_idx[mask_else_small]
            idx_left[inds] = 0
            idx_right[inds] = 0
            theta[inds] = 1.0

        # ---- w > v[-1] ----
        mask_cond_large = (w_div_v_last >= 0) & (w_div_v_last <= 1) & edge_large
        mask_else_large = edge_large & (~mask_cond_large)

        if mask_cond_large.any():
            inds = edge_idx[mask_cond_large]
            w_large = w_edge[mask_cond_large].unsqueeze(1)          # (k,1)
            div_mat = w_large / v.unsqueeze(0)                       # (k,C)
            val_mat = div_mat * xi.unsqueeze(0)                      # (k,C)
            i_min = torch.argmin(val_mat, dim=1)                     # (k,)
            vals_min = div_mat[torch.arange(i_min.shape[0], device=device), i_min]

            idx_left[inds] = i_min
            idx_right[inds] = i_min
            theta[inds] = vals_min

        if mask_else_large.any():
            inds = edge_idx[mask_else_large]
            idx_left[inds] = C - 1
            idx_right[inds] = C - 1
            theta[inds] = 1.0

    # === Step 6: Intermediate case (same as dense: compare method1 vs method2) ===
    if mask_mid.any():
        mid_idx = torch.nonzero(mask_mid, as_tuple=True)[0]
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # ---- First method (dense builds x1_sol; we compute obj1 directly) ----
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector] == 1).unsqueeze(0)

        valid_i0 = torch.where(
            valid,
            torch.arange(C, device=device)[None, :],
            torch.tensor(float("inf"), device=device)
        )
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]             # (M_mid,)
        v_i0 = v[i0]

        theta1 = w_mid / v_i0             # (M_mid,)
        obj1 = theta1 * xi[i0]            # since x has only one nonzero: theta1 at i0
        obj1 = torch.where(theta1 < 0, torch.tensor(float("inf"), device=device), obj1)

        # ---- Second method (dense builds x2_sol; we compute obj2 directly) ----
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0].to(device=device, dtype=torch.long)
        i_right = torch.searchsorted(v[one_indices], w_mid, right=False)
        i_right = i_right.clamp(min=1, max=one_indices.shape[0] - 1)

        idx_r = one_indices[i_right]
        idx_l = one_indices[i_right - 1]

        v_left = v[idx_l]
        v_right = v[idx_r]
        theta2 = (w_mid - v_right) / (v_left - v_right + 1e-8)  # same epsilon as dense

        obj2 = theta2 * xi[idx_l] + (1.0 - theta2) * xi[idx_r]

        # ---- Choose better (same criterion as dense) ----
        better_first = obj1 < obj2

        idx_left[mid_idx] = torch.where(better_first, i0, idx_l)
        idx_right[mid_idx] = torch.where(better_first, i0, idx_r)
        theta[mid_idx] = torch.where(better_first, theta1, theta2)

    # === Step 7: Compute lambda_opt (same as dense) ===
    denominator = v[idx_right] - v[idx_left]
    denominator_zero_mask = denominator == 0

    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = - xi / v
    lambda_opt_zero = lambda_opt_zero_full[idx_left]

    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # === Step 8: Objective (same as dense, but without x @ xi) ===
    objective_values = objective_constant + theta * xi[idx_left]

    mask_diff = idx_right != idx_left
    if mask_diff.any():
        objective_values[mask_diff] += (
            (1.0 - theta[mask_diff]) * xi[idx_right[mask_diff]]
        )

    # === Step 9: Placeholder "x" to keep signature ===
    x_placeholder = torch.stack(
        [idx_left.to(torch.int32), idx_right.to(torch.int32), theta],
        dim=1
    )  # (M, 3)

    # (No aggressive cleanup here: identical results are the goal; freeing is caller’s choice)
    return x_placeholder, lambda_opt, objective_values

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

