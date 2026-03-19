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

""" USED TO MAKE A DIAGNOSTIC PRINT IN test_accuracyGPU, TO CHECK IF THE TIME WAS SPENT WAITING FOR THE DATALOADER OR ON THE GPU  """
@torch.no_grad()
def test_accuracyGPU(model, dataloader, device):
    """
    Fast accuracy + timing breakdown:
    - data_wait_time: time spent waiting for the next batch from the dataloader
    - h2d_time: transfer time host -> device
    - forward_time: forward pass time
    """
    import time
    import torch
    import torch.distributed as dist

    was_training = model.training
    model.eval()

    correct = torch.zeros((), device=device, dtype=torch.long)
    total   = torch.zeros((), device=device, dtype=torch.long)

    use_cuda = (device.type == "cuda")
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    data_wait_time = 0.0
    h2d_time = 0.0
    forward_time = 0.0
    num_batches = 0

    autocast_ctx = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_cuda,
    )

    it = iter(dataloader)
    
    seen_local = 0

    with torch.inference_mode(), autocast_ctx:
        while True:
            # -------------------------
            # 1) Wait for data (host) --- this is the "hidden" 
            # part of the dataloader time, which includes CPU 
            # processing and potential waiting for the next 
            # batch to be ready
            # -------------------------
            t0 = time.time()
            try:
                images, labels = next(it)
            except StopIteration:
                break
            data_wait_time += time.time() - t0

            # -------------------------
            # 2) Host -> Device
            # -------------------------
            t1 = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            seen_local += labels.numel()

            if use_cuda:
                torch.cuda.synchronize(device)
            h2d_time += time.time() - t1

            # -------------------------
            # 3) Forward GPU
            # -------------------------
            t2 = time.time()
            if use_cuda:
                images = images.contiguous(memory_format=torch.channels_last)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum()
            total   += labels.numel()

            if use_cuda:
                torch.cuda.synchronize(device)
            forward_time += time.time() - t2

            num_batches += 1

    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total,   op=dist.ReduceOp.SUM)

    seen_total = torch.tensor(seen_local, device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(seen_total, op=dist.ReduceOp.SUM)        

    acc = (correct.float() * 100.0 / total.float()).item() if total.item() > 0 else 0.0

    total_measured = data_wait_time + h2d_time + forward_time

    if local_rank == 0:
        print(
            f"[test_accuracyGPU] batches={num_batches} | seen_local={seen_local} | "
            f"seen_total={seen_total.item()} | data_wait={data_wait_time:.2f}s | "
            f"h2d={h2d_time:.2f}s | forward={forward_time:.2f}s | "
            f"measured_total={total_measured:.2f}s",
            flush=True
        )

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
        # Solve the simil-knapsack problem for the current xi
        if(pruning == "Y"):
            if(device.type == "cuda"):
                #x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning_sparse(xi, v, w, C, device, delta)
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning_sparse_leonardo(xi, v, w, C, device, delta)
            else:
                x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(xi, v, w, C, device, delta)
        elif(pruning == "N"):
            x_i_star, lambda_plus, phi_plus = knapsack_specialized(xi, v, w, C, device)

        # === Step: sum_x_star ===
        # Old behavior: x_i_star is dense (M, C) -> sum along rows
        # New memory-light behavior: x_i_star is (M, 3) = [idx_left, idx_right, theta]
        if x_i_star.dim() == 2 and x_i_star.size(1) == 3:
            # === Memory-light sum (NO dense x) ===
            idx_left = x_i_star[:, 0].to(dtype=torch.long, device=device)
            idx_right = x_i_star[:, 1].to(dtype=torch.long, device=device)
            theta = x_i_star[:, 2].to(dtype=torch.float32, device=device)

            sum_x_star = torch.zeros(C, dtype=torch.float32, device=device)

            # Add theta contribution on idx_left
            sum_x_star.scatter_add_(0, idx_left, theta)

            # Add (1-theta) contribution on idx_right
            # BUT: if idx_left == idx_right (1-sparse case), we must not double count
            mask_diff = idx_right != idx_left
            if mask_diff.any():
                sum_x_star.scatter_add_(0, idx_right[mask_diff], (1.0 - theta[mask_diff]))

        else:
            # === Dense sum ===
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
