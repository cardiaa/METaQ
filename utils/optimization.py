import torch  
from torch.linalg import norm  
from .knapsack import knapsack_specialized  
from .knapsack import knapsack_specialized_pruning

def FISTA(xi, v, w, C, delta, subgradient_step, device, max_iterations):
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
    
    upper_c = w.size(0)  # Define an upper bound for constraints
    
    # Initialize previous values for FISTA acceleration
    xi_prev = xi.clone().to(device)
    t_prev = torch.tensor(1.0, device=device)

    for iteration in range(1, max_iterations + 1):
        # Solve the simil-knapsack problem for the current xi
        x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(xi, v, w, C, delta, device)
        sum_x_star = torch.sum(x_i_star, dim=0)

        # Compute the optimal c values c_star
        c_star = torch.exp(torch.log(torch.tensor(2.0, device=device)) * xi - 1)
        c_star = torch.clamp(c_star, min=0, max=upper_c)

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

    return xi, lambda_plus, x_i_star, phi


def ProximalBM(xi, v, w, C, delta, zeta, subgradient_step, device, max_iterations):
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
    
    upper_c = w.size(0)  # Define an upper bound for constraints
    
    # Parameters for the bundle method
    epsilon = 1e-5  # Convergence tolerance
    bundle_size = 5  # Maximum bundle size
    bundle = []  # Initialize the bundle

    for iteration in range(1, max_iterations + 1):
        # Solve the knapsack problem for the current xi
        x_i_star, lambda_plus, phi_plus = knapsack_specialized_pruning(xi, v, w, C, device)
        sum_x_star = torch.sum(x_i_star, dim=0)

        # Compute the optimal c values c_star
        c_star = torch.exp(torch.log(torch.tensor(2.0, device=device)) * xi - 1)
        c_star = torch.clamp(c_star, min=0, max=upper_c)

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

    return xi, lambda_plus, x_i_star, phi
