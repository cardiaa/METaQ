import torch  
from torch.linalg import norm  
from .knapsack import knapsack_specialized, knapsack_specialized_pruning, knapsack_specialized_pruning_sparse, knapsack_specialized_pruning_sparse_leonardo

def test_accuracy(model, dataloader, device):
    """
    Function to calculate the accuracy of a model on a given dataloader.
    """
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
    
    accuracy = 100 * correct / total  # Compute accuracy percentage
    return accuracy

''' TO UNCOMMENT OUT OF TESTING
@torch.no_grad()
def test_accuracyGPU(model, dataloader, device):
    """
    Fast accuracy:
    - counters on GPU (no sync per batch)
    - inference_mode
    - autocast bf16 on CUDA (A100-friendly)
    - optional channels_last for images (often helps conv nets)
    """
    was_training = model.training
    model.eval()

    correct = torch.zeros((), device=device, dtype=torch.long)
    total   = torch.zeros((), device=device, dtype=torch.long)

    use_cuda = (device.type == "cuda")
    autocast_ctx = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_cuda,
    )

    with torch.inference_mode(), autocast_ctx:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # helps convs (AlexNet/VGG) in many cases
            if use_cuda:
                images = images.contiguous(memory_format=torch.channels_last)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum()
            total   += labels.numel()

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total,   op=torch.distributed.ReduceOp.SUM)

    # one final sync (inevitable to print/return a Python float)
    acc = (correct.float() * 100.0 / total.float()).item() if total.item() > 0 else 0.0

    if was_training:
        model.train()
    return acc
'''

@torch.no_grad()
def test_accuracyGPU(model, dataloader, device):
    """
    Fast distributed accuracy:
    - GPU counters, with one all-reduce at the end
    - inference_mode/autocast bf16 on CUDA
    - no per-batch debug prints or timing synchronizations
    """
    import torch.distributed as dist

    was_training = model.training
    model.eval()

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)

    use_cuda = (device.type == "cuda")
    autocast_ctx = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_cuda,
    )

    with torch.inference_mode(), autocast_ctx:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_cuda:
                images = images.contiguous(memory_format=torch.channels_last)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum()
            total += labels.numel()

    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    acc = (correct.float() * 100.0 / total.float()).item() if total.item() > 0 else 0.0

    if was_training:
        model.train()

    return acc

def FISTA(xi, v, w, C, upper_c, lower_c, delta, subgradient_step, device, max_iterations, pruning):
    """
    Implements the Fast Iterative Shrinking-Thresholding Algorithm (FISTA) 
    for optimizing a constrained objective function.

    Args:
        xi (torch.Tensor): Initial parameter vector.
        v (torch.Tensor): Constraint-related vector.
        w (torch.Tensor): Weight vector.
        C (float): Constraint parameter.
        subgradient_step (float): Step size for subgradient descent.
        max_iterations (int): Maximum number of iterations.

    Returns:
        tuple: Updated xi, lambda_plus (Lagrange multiplier), 
               x_i_star (optimal allocation), and phi (objective function value).
    """
    
    # Initialize previous values for FISTA acceleration
    xi_prev = xi.clone().to(device)
    t_prev = torch.tensor(1.0, device=device)

    for iteration in range(1, max_iterations + 1):
        #print(f"FISTA's Iteration {iteration}", flush=True)
        # Solve the simil-knapsack problem for the current xi
        if(pruning == "Y"):
            #print("outside function", flush=True)
            if(device.type == "cuda"):
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning_sparse(xi, v, w, C, device, delta)
            else:
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(xi, v, w, C, device, delta)
        elif(pruning == "N"):
            x_i_star, lambda_plus, phi_plus = knapsack_specialized(xi, v, w, C, device)
        
        #print(f"B FISTA's Iteration {iteration}", flush=True)

        sum_x_star = torch.sum(x_i_star, dim=0)

        # Compute the optimal c values c_star
        c_star = torch.exp(torch.log(torch.tensor(2.0, device=device)) * xi - 1)
        c_star = torch.clamp(c_star, min=lower_c, max=upper_c)

        # Compute the super-gradient
        g = -(c_star - sum_x_star)

        # Compute the objective function value phi
        phi1 = torch.sum(c_star * torch.log(c_star) / torch.log(torch.tensor(2.0, device=device)))
        phi2 = -torch.sum(xi * c_star)
        phi3 = torch.sum(xi * sum_x_star)
        phi = phi1 + phi2 + phi3

        # FISTA acceleration step
        t_current = (1 + torch.sqrt(1 + 4 * t_prev**2)) / 2
        y = xi + ((t_prev - 1) / t_current) * (xi - xi_prev)

        # Gradient update step
        xi_next = y + (1 / subgradient_step) * g 

        # Update variables for next iteration
        xi_prev = xi.clone()
        xi = xi_next.clone()
        t_prev = t_current

        # Ensure xi remains sorted
        xi = torch.sort(xi)[0]

    #return xi, lambda_plus, x_i_star, phi
    return xi, lambda_plus

def FISTA_leonardo(xi, v, w, C, upper_c, lower_c, delta, subgradient_step, device, max_iterations, pruning):
    """
    Sparse-aware FISTA for the METaQ entropy dual.

    If pruning == "Y" and xi has length C + 1:
        xi[0]  = multiplier for the explicit zero/pruning symbol
        xi[1:] = multipliers for the C non-zero quantization buckets

    The quantization vector v still has length C and contains only non-zero
    levels.  The zero symbol is represented by the residual mass

        z_i = 1 - sum_b x_{i,b}

    and is included in the entropy dual through its own multiplier xi[0].
    """

    xi = xi.to(dtype=torch.float32, device=device)
    xi_prev = xi.clone().to(device)
    t_prev = torch.tensor(1.0, device=device)

    use_zero_symbol = (pruning == "Y" and xi.numel() == C + 1)

    for iteration in range(1, max_iterations + 1):
        # Solve the per-weight sparse knapsack.
        # In the sparse-aware case, xi has length C+1; the knapsack function
        # internally converts it to the effective C-dimensional cost vector:
        #
        #     xi_b - xi_zero - delta
        #
        if pruning == "Y":
            if device.type == "cuda":
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning_sparse_leonardo(
                    xi,
                    v,
                    w,
                    C,
                    device,
                    delta,
                )
            else:
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(
                    xi,
                    v,
                    w,
                    C,
                    device,
                    delta,
                )
        elif pruning == "N":
            x_i_star, lambda_plus, phi_plus = knapsack_specialized(
                xi,
                v,
                w,
                C,
                device,
            )
        else:
            raise ValueError(f"Unsupported pruning flag: {pruning}")

        # ------------------------------------------------------------------
        # Recover bucket occupancies.
        #
        # Memory-light Leonardo path:
        #     x_i_star is (M, 3) = [idx_left, idx_right, theta]
        #
        # Dense path:
        #     x_i_star is (M, C)
        # ------------------------------------------------------------------
        if x_i_star.dim() == 2 and x_i_star.size(1) == 3:
            idx_left = x_i_star[:, 0].to(dtype=torch.long, device=device)
            idx_right = x_i_star[:, 1].to(dtype=torch.long, device=device)
            theta = x_i_star[:, 2].to(dtype=torch.float32, device=device)

            with torch.no_grad():
                sum_x_per_weight = theta.clone()
                mask_diff_debug = idx_right != idx_left

                if mask_diff_debug.any():
                    sum_x_per_weight[mask_diff_debug] += 1.0 - theta[mask_diff_debug]

                violation = 1.0 - sum_x_per_weight

                if not hasattr(FISTA_leonardo, "_delta_debug"):
                    FISTA_leonardo._delta_debug = {
                        "calls": 0,
                        "mean_sum_x": 0.0,
                        "mean_violation": 0.0,
                        "frac_null_x": 0.0,
                        "frac_sum_x_lt_0_5": 0.0,
                        "frac_two_bucket": 0.0,

                        # Sparse-aware zero-symbol diagnostics.
                        "xi_zero": 0.0,
                        "xi_bucket_mean": 0.0,
                        "xi_bucket_min": 0.0,
                        "xi_bucket_max": 0.0,
                        "effective_xi_mean": 0.0,
                        "effective_xi_min": 0.0,
                        "effective_xi_max": 0.0,
                        "objective_constant": 0.0,
                        "sum_z_per_weight": 0.0,
                        "c_zero_per_weight": 0.0,
                        "g_zero_per_weight": 0.0,
                    }

                dbg = FISTA_leonardo._delta_debug
                dbg["calls"] += 1
                dbg["mean_sum_x"] += sum_x_per_weight.mean().item()
                dbg["mean_violation"] += violation.mean().item()
                dbg["frac_null_x"] += (sum_x_per_weight < 1e-6).float().mean().item()
                dbg["frac_sum_x_lt_0_5"] += (sum_x_per_weight < 0.5).float().mean().item()
                dbg["frac_two_bucket"] += mask_diff_debug.float().mean().item()

            sum_x_star = torch.zeros(C, dtype=torch.float32, device=device)

            # Add theta contribution on idx_left.
            sum_x_star.scatter_add_(0, idx_left, theta)

            # Add (1-theta) contribution on idx_right.
            # If idx_left == idx_right, do not double-count.
            mask_diff = idx_right != idx_left
            if mask_diff.any():
                sum_x_star.scatter_add_(
                    0,
                    idx_right[mask_diff],
                    1.0 - theta[mask_diff],
                )

            # Residual mass assigned to the zero/pruning symbol.
            sum_z_star = (1.0 - sum_x_per_weight).sum().reshape(1)

        else:
            sum_x_star = torch.sum(x_i_star, dim=0)

            if use_zero_symbol:
                sum_z_star = (
                    torch.as_tensor(w.numel(), dtype=torch.float32, device=device)
                    - sum_x_star.sum()
                ).reshape(1)

        # ------------------------------------------------------------------
        # Compute optimal c values and the dual supergradient.
        # ------------------------------------------------------------------
        log2_t = torch.log(torch.tensor(2.0, device=device))

        if use_zero_symbol:
            # xi has length C+1:
            #     xi[0]  -> zero symbol
            #     xi[1:] -> non-zero buckets
            sum_symbol_star = torch.cat([sum_z_star, sum_x_star])

            c_star = torch.exp(log2_t * xi - 1)
            c_star = torch.clamp(c_star, min=lower_c, max=upper_c)

            g = -(c_star - sum_symbol_star)

            phi1 = torch.sum(c_star * torch.log(c_star) / log2_t)
            phi2 = -torch.sum(xi * c_star)
            phi3 = torch.sum(xi * sum_symbol_star)
            phi = phi1 + phi2 + phi3

            with torch.no_grad():
                if hasattr(FISTA_leonardo, "_delta_debug"):
                    dbg = FISTA_leonardo._delta_debug

                    xi_zero_dbg = xi[0].detach()
                    xi_buckets_dbg = xi[1:].detach()

                    delta_t_dbg = torch.as_tensor(delta, dtype=torch.float32, device=device)
                    effective_xi_dbg = xi_buckets_dbg - xi_zero_dbg - delta_t_dbg
                    objective_constant_dbg = xi_zero_dbg + delta_t_dbg

                    n_weights_dbg = float(w.numel())

                    dbg["xi_zero"] += xi_zero_dbg.item()
                    dbg["xi_bucket_mean"] += xi_buckets_dbg.mean().item()
                    dbg["xi_bucket_min"] += xi_buckets_dbg.min().item()
                    dbg["xi_bucket_max"] += xi_buckets_dbg.max().item()

                    dbg["effective_xi_mean"] += effective_xi_dbg.mean().item()
                    dbg["effective_xi_min"] += effective_xi_dbg.min().item()
                    dbg["effective_xi_max"] += effective_xi_dbg.max().item()

                    dbg["objective_constant"] += objective_constant_dbg.item()

                    dbg["sum_z_per_weight"] += (sum_z_star / n_weights_dbg).item()
                    dbg["c_zero_per_weight"] += (c_star[0] / n_weights_dbg).item()
                    dbg["g_zero_per_weight"] += (g[0] / n_weights_dbg).item()            

        else:
            c_star = torch.exp(log2_t * xi - 1)
            c_star = torch.clamp(c_star, min=lower_c, max=upper_c)

            g = -(c_star - sum_x_star)

            phi1 = torch.sum(c_star * torch.log(c_star) / log2_t)
            phi2 = -torch.sum(xi * c_star)
            phi3 = torch.sum(xi * sum_x_star)
            phi = phi1 + phi2 + phi3

        # ------------------------------------------------------------------
        # FISTA acceleration step.
        # ------------------------------------------------------------------
        t_current = (1 + torch.sqrt(1 + 4 * t_prev**2)) / 2
        y = xi + ((t_prev - 1) / t_current) * (xi - xi_prev)

        xi_next = y + (1 / subgradient_step) * g

        xi_prev = xi.clone()
        xi = xi_next.clone()
        t_prev = t_current

        # Keep bucket multipliers ordered, but do NOT sort the zero multiplier
        # together with the non-zero bucket multipliers.
        if use_zero_symbol:
            xi_zero = xi[:1]
            xi_buckets = torch.sort(xi[1:])[0]
            xi = torch.cat([xi_zero, xi_buckets])
        else:
            xi = torch.sort(xi)[0]

    return xi, lambda_plus

def ProximalBM(xi, v, w, C, upper_c, lower_c, delta, zeta, subgradient_step, device, max_iterations, pruning):
    """
    Implements the Proximal Bundle Method (PBM) for solving constrained 
    optimization problems using bundle techniques.

    Args:
        xi (torch.Tensor): Initial parameter vector.
        v (torch.Tensor): Constraint-related vector.
        w (torch.Tensor): Weight vector.
        C (float): Constraint parameter.
        zeta (float): Regularization parameter for proximal term.
        subgradient_step (float): Step size for subgradient descent.
        max_iterations (int): Maximum number of iterations.

    Returns:
        tuple: Updated xi, lambda_plus (Lagrange multiplier), 
               x_i_star (optimal allocation), and phi (objective function value).
    """
    
    # Parameters for the bundle method
    epsilon = 1e-5  # Convergence tolerance
    bundle_size = 5  # Maximum bundle size
    bundle = []  # Initialize the bundle

    for iteration in range(1, max_iterations + 1):
        # Solve the knapsack problem for the current xi
        if(pruning == "Y"):
            if(device.type == "cuda"):
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning_sparse(xi, v, w, C, device, delta)
            else:
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(xi, v, w, C, device, delta)
        elif(pruning == "N"):
            x_i_star, lambda_plus, phi_plus = knapsack_specialized(xi, v, w, C, device)

        sum_x_star = torch.sum(x_i_star, dim=0)

        # Compute the optimal c values c_star
        c_star = torch.exp(torch.log(torch.tensor(2.0, device=device)) * xi - 1)
        c_star = torch.clamp(c_star, min=lower_c, max=upper_c)

        # Compute the super-gradient
        g = -(c_star - sum_x_star)

        # Compute the objective function value phi
        phi1 = torch.sum(c_star * torch.log(c_star) / torch.log(torch.tensor(2.0, device=device)))
        phi2 = -torch.sum(xi * c_star)
        phi3 = torch.sum(xi * sum_x_star)
        phi = phi1 + phi2 + phi3

        # Add the current point to the bundle
        bundle.append((xi.clone().to(device), phi, g.clone().to(device)))
        if len(bundle) > bundle_size:
            bundle.pop(0)

        # Solve the quadratic regularization subproblem
        bundle_points = torch.stack([item[0] for item in bundle])
        bundle_phis = torch.tensor([item[1] for item in bundle], device=device)
        bundle_gradients = torch.stack([item[2] for item in bundle])

        # Construct the quadratic approximation model
        diff = xi - bundle_points
        model_phi = bundle_phis + torch.sum(bundle_gradients * diff, dim=1)
        proximal_term = (zeta / 2) * norm(diff, dim=1)**2
        subproblem_objective = model_phi + proximal_term

        # Determine the next xi by minimizing the subproblem objective
        best_idx = torch.argmax(subproblem_objective)
        xi_next = bundle_points[best_idx] + (1 / zeta) * bundle_gradients[best_idx]

        # Clip xi to enforce constraints
        xi_next = torch.clamp(xi_next, min=0.01, max=upper_c)

        # Check for convergence
        if norm(xi_next - xi) < epsilon:
            break

        # Update xi for the next iteration
        xi = xi_next.clone().to(device)

    #return xi, lambda_plus, x_i_star, phi
    return xi, lambda_plus
