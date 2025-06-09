import torch   

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
    xi = xi.to(device) - delta
    v = v.to(device)
    w = w.to(device)

    M = w.shape[0]

    # Step 1: Compute x_plus
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
    x_plus[torch.tensor(b_list, device=device)] = 1

    # Step 2: Precompute
    ratio = xi / v
    neg_indices = torch.where(v < 0)[0]
    pos_indices = torch.where(v >= 0)[0]
    neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
    pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
    b_vector = torch.cat([neg_sorted, pos_sorted], dim=0)

    # Step 3: Masks
    mask_small = w < v[0]
    mask_large = w > v[-1]
    mask_mid = (~mask_small) & (~mask_large)
    mask_edge = mask_small | mask_large

    # Step 4: Init outputs
    x = torch.zeros(M, C, device=device)
    lambda_opt = torch.zeros(M, device=device)
    idx_left = torch.zeros_like(w, dtype=torch.long)
    idx_right = torch.zeros_like(w, dtype=torch.long)

    # Step 5: Edge cases (versione aggiornata)
    if mask_edge.any():
        w_edge = w[mask_edge]  # (M_edge,)
        M_edge = w_edge.shape[0]

        # Inizializza soluzione edge
        x_edge = torch.zeros((M_edge, C), device=device)
        i_min = torch.full((M_edge,), -1, dtype=torch.long, device=device)

        # Calcola rapporto w/v
        ratio_edge = w_edge.unsqueeze(1) / v.unsqueeze(0)  # (M_edge, C)
        valid = (ratio_edge >= 0) & (ratio_edge <= 1)  # condizioni valide

        # Punteggio per ogni item: w/v * xi
        scores = ratio_edge * xi.unsqueeze(0)  # (M_edge, C)
        scores[~valid] = float('inf')  # ignora valori non validi

        # Trova indice del minimo punteggio valido, se esiste
        valid_any = valid.any(dim=1)  # (M_edge,)
        valid_rows = valid_any.nonzero(as_tuple=True)[0]
        if valid_rows.numel() > 0:
            min_scores, min_indices = torch.min(scores[valid_rows], dim=1)
            i_min[valid_rows] = min_indices
            chosen_ratios = ratio_edge[valid_rows, min_indices]
            x_edge[valid_rows, min_indices] = chosen_ratios

        # Fallback dove non esiste nessun valore valido
        fallback_mask = ~valid_any
        fallback_rows = torch.arange(M_edge, device=device)[fallback_mask]
        fallback_w = w_edge[fallback_mask]

        fallback_first = fallback_w < v[0]
        fallback_last = fallback_w > v[-1]

        x_edge[fallback_rows[fallback_first], 0] = 1.0
        x_edge[fallback_rows[fallback_last], C - 1] = 1.0

        # Assegna soluzione finale a x
        x[mask_edge] = x_edge

        # Aggiorna idx_left e idx_right
        full_idx = torch.arange(M, device=device)[mask_edge]
        i_min_fallback = i_min.clone()
        i_min_fallback[fallback_mask][fallback_first] = 0
        i_min_fallback[fallback_mask][fallback_last] = C - 1
        i_min_fallback[i_min_fallback == -1] = 0  # fallback di sicurezza
        idx_left[full_idx] = i_min_fallback
        idx_right[full_idx] = i_min_fallback


    # Step 6: Intermediate Case
    if mask_mid.any():
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # First method
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector].unsqueeze(0).bool())
        valid_i0 = torch.where(valid, torch.arange(C, device=device)[None, :], float('inf'))
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]
        theta1 = w_mid / v_i0
        x1_sol = torch.zeros(M_mid, C, device=device)
        x1_sol[torch.arange(M_mid), i0] = theta1
        obj1 = x1_sol @ xi
        obj1[theta1 < 0] = float('inf')

        # Second method
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0]
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

        # Update idx_left and idx_right
        full_idx = torch.arange(M, device=device)[mask_mid]
        idx_left[full_idx] = torch.where(better_first, i0, idx_left_mid)
        idx_right[full_idx] = torch.where(better_first, i0, idx_right_mid)

    # Step 7: Compute lambda_opt
    denominator = v[idx_right] - v[idx_left]
    denominator_zero_mask = denominator.abs() < 1e-8

    lambda_opt_zero_full = -(xi + delta) / v
    lambda_opt_zero = lambda_opt_zero_full[idx_left]
    lambda_opt_nonzero = -(xi[idx_right] - xi[idx_left]) / (denominator + 1e-8)
    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # Step 8: Objective
    objective_values = delta + x @ xi

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

