import torch  
import numpy as np  

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

    lambda_opt_nonzero = (xi[idx_right] - xi[idx_left]) / denominator
    lambda_opt_zero_full = xi / v
    lambda_opt_zero_full[0] = 0
    lambda_opt_zero = lambda_opt_zero_full[idx_left]

    lambda_opt = torch.where(denominator_zero_mask, lambda_opt_zero, lambda_opt_nonzero)

    # Compute objective function values
    objective_values = torch.matmul(x_opt, xi)

    return x_opt, lambda_opt, objective_values

def knapsack_specialized_pruning(xi, v, w, C, delta, device):
    xi = xi.to(device)
    v = v.to(device)
    w = w.to(device)

    xi = xi - delta

    M = w.shape[0]

    # Step 1: Calcolo x_plus (breakpoints)
    b_list = []
    b = 0
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
    x_plus[torch.tensor(b_list)] = 1

    # Preallocazioni
    x = torch.zeros(M, C)
    lambda_opt = torch.zeros(M)

    # Step 2: Classificazione dei problemi
    v0 = v[0]
    v_last = v[-1]
    mask_small = w < v0
    mask_large = w > v_last
    mask_mid = (~mask_small) & (~mask_large)

    # CASO: w > v[-1]
    x[mask_large, -1] = 1

    # CASO: w < v[0]
    x[mask_small, 0] = 1

    # CASO INTERMEDIO
    if mask_mid.any():
        M_mid = mask_mid.sum() #Numero di v[0]<w<v[-1]
        w_mid = w[mask_mid]
        ratio = xi / v
        neg_indices = torch.where(ratio < 0)[0]
        neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
        pos_indices = torch.where(ratio >= 0)[0]
        pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
        b_vector = torch.cat([neg_sorted, pos_sorted], dim=0)
        ratio_b = w_mid[:, None] / v[b_vector]
        x_plus_b = x_plus[b_vector].bool()
        cond1 = (ratio_b >= 0) & x_plus_b
        valid_i0 = cond1.float() * torch.arange(C)[None, :]
        valid_i0[~cond1] = float('inf')
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]
        x_single = w_mid / v_i0
        invalid_i0 = x_plus[i0] == 0
        use_two = x_single > 1
        i1 = torch.full_like(i0, fill_value=-1)
        if use_two.any():
            b_vector_exp = b_vector.unsqueeze(0).expand(M_mid, -1)
            i0_exp = i0.unsqueeze(1).expand_as(b_vector_exp)
            x_plus_mask = x_plus[b_vector_exp] == 1
            greater_mask = b_vector_exp > i0_exp
            valid_mask = x_plus_mask & greater_mask
            masked_b_vector = torch.where(valid_mask, b_vector_exp, torch.full_like(b_vector_exp, C))
            i1_candidate, _ = masked_b_vector.min(dim=1)
            i1_candidate_use_two = i1_candidate[use_two]
            valid_i1_use_two = i1_candidate_use_two < C
            i0_use_two = i0[use_two]
            i1[use_two] = torch.where(valid_i1_use_two, i1_candidate_use_two, i0_use_two)

        # Costruzione x_mid
        x_mid = torch.zeros(M_mid, C)

        # Caso: uso un solo indice
        mask_one = ~use_two
        rows_one = torch.where(mask_one)[0]
        cols_one = i0[mask_one]
        x_mid[rows_one, cols_one] = torch.clamp(torch.round(w_mid[mask_one] / v[cols_one], decimals=5), 0.0, 1.0)

        # Caso: combinazione convessa
        mask_two = use_two & (i1 != i0)
        rows_two = torch.where(mask_two)[0]
        idx0 = i0[mask_two]
        idx1 = i1[mask_two]
        v0 = v[idx0]
        v1 = v[idx1]
        w_sel = w_mid[mask_two]
        theta = (w_sel - v1) / (v0 - v1)
        x_mid[rows_two, idx0] = torch.round(theta, decimals=5)
        x_mid[rows_two, idx1] = torch.round(1 - theta, decimals=5)

        x[mask_mid] = x_mid
        
    # === Calcolo vectorizzato dei moltiplicatori ===
    eps = 1e-6
    nz_mask = torch.abs(x) > eps
    nz_counts = nz_mask.sum(dim=1)
    lambda_opt = torch.zeros(x.shape[0])

    # Caso 1 valore non nullo
    m1 = torch.where(nz_counts == 1)[0]
    if m1.numel() > 0:
        submask = nz_mask[m1]
        indices = submask.nonzero(as_tuple=False) 
        i = indices[:, 1]
        lambda_opt[m1] = -xi[i] / v[i]
        lambda_opt[m1] = torch.round(lambda_opt[m1], decimals=5)

    # Caso 2 valori non nulli
    m2 = torch.where(nz_counts == 2)[0]
    if m2.numel() > 0:
        indices = nz_mask[m2].nonzero().reshape(-1, 2)
        grouped = indices.view(-1, 2, 2)
        i = grouped[:, 0, 1]
        j = grouped[:, 1, 1]
        delta_xi = xi[j] - xi[i]
        delta_idx = j - i
        passo = v[1] - v[0]
        lambda_opt[m2] = -delta_xi / (delta_idx * passo)
        lambda_opt[m2] = torch.round(lambda_opt[m2], decimals=5)

    objective_values = delta + x @ xi
    
    return x, lambda_opt, objective_values

def knapsack_specialized_histo(xi, v, w, C, device):
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

