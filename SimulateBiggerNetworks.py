import torch
import numpy as np
import time

def knapsack_specialized_pruning_complete_parallel(xi, v, w, C):
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

    # === Step 2: Precompute ===
    ratio = xi / v
    neg_indices = torch.where(v < 0)[0]
    pos_indices = torch.where(v >= 0)[0]
    neg_sorted = neg_indices[torch.argsort(ratio[neg_indices], descending=True)]
    pos_sorted = pos_indices[torch.argsort(ratio[pos_indices])]
    b_vector = torch.cat([neg_sorted, pos_sorted], dim=0)

    # === Step 3: Masks ===
    mask_small = w < v[0]
    mask_large = w > v[-1]
    mask_mid = (~mask_small) & (~mask_large)
    mask_edge = (mask_small | mask_large)

    # === Step 4: Initialize outputs ===
    x = torch.zeros(M, C)
    lambda_opt = torch.zeros(M)

    # === Step 5: Edge cases ===
    if mask_edge.any():
        w_edge = w[mask_edge]
        condition = (w_edge / v[0] >= 0) & (w_edge / v[0] <= 1)

        x_edge = torch.zeros((w_edge.shape[0], C))

        if condition.any():
            valid_w = w_edge[condition]
            obj = (valid_w[:, None] / v) * xi  # [M_valid, C]
            obj[:, x_plus == 0] = float('inf')  # only x_plus = 1 allowed
            best_idx = torch.argmin(obj, dim=1)
            theta = valid_w / v[best_idx]
            x_edge[condition, best_idx] = theta

        x[mask_edge] = x_edge

    # === Step 6: Intermediate Case ===
    if mask_mid.any():
        w_mid = w[mask_mid]
        M_mid = w_mid.shape[0]

        # --- First Method ---
        ratio_b = w_mid[:, None] / v[b_vector]
        valid = (ratio_b >= 0) & (ratio_b <= 1) & (x_plus[b_vector] == 1).unsqueeze(0)
        valid_i0 = torch.where(valid, torch.arange(C)[None, :], float('inf'))
        i0_pos = valid_i0.argmin(dim=1)
        i0 = b_vector[i0_pos]
        v_i0 = v[i0]
        x1_sol = torch.zeros(M_mid, C)
        theta1 = w_mid / v_i0
        x1_sol[torch.arange(M_mid), i0] = theta1
        obj1 = x1_sol @ xi
        obj1[theta1 < 0] = float('inf')

        # --- Second Method ---
        one_indices = torch.nonzero(x_plus, as_tuple=True)[0]
        i_right = torch.searchsorted(v[one_indices], w_mid, right=False)
        i_right = i_right.clamp(min=1, max=one_indices.shape[0] - 1)
        idx_right = one_indices[i_right]
        idx_left = one_indices[i_right - 1]

        v_left = v[idx_left]
        v_right = v[idx_right]
        theta2 = (w_mid - v_right) / (v_left - v_right + 1e-8)

        x2_sol = torch.zeros(M_mid, C)
        x2_sol[torch.arange(M_mid), idx_left] = theta2
        x2_sol[torch.arange(M_mid), idx_right] = 1 - theta2
        obj2 = x2_sol @ xi

        # --- Choose better ---
        better_first = obj1 < obj2
        final_x = torch.where(better_first.unsqueeze(1), x1_sol, x2_sol)
        x[mask_mid] = final_x

    # === Step 7: Compute lambda_opt ===
    eps = 1e-6
    nz_mask = torch.abs(x) > eps
    nz_counts = nz_mask.sum(dim=1)

    # One non-zero variable
    m1 = torch.where(nz_counts == 1)[0]
    if m1.numel() > 0:
        indices = nz_mask[m1].nonzero(as_tuple=False)
        i = indices[:, 1]
        lambda_opt[m1] = -xi[i] / v[i]
        lambda_opt[m1] = torch.round(lambda_opt[m1], decimals=5)

    # Two non-zero variables
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

    # === Step 8: Objective ===
    objective_values = x @ xi

    return x, lambda_opt, objective_values, x_plus

C = 128
M_LeNet5 = 44000
M_LeNet100 = int(1.5e5)
M_LeNet300 = int(3e5)
M_AlexNet = int(6e7)
M_VGG16 = int(1.38e8)
N = 10

w0 = -0.01
r = 1.32
a = -2
b = 2.2
min_w, max_w = w0 - r, w0 + r

np.random.seed(41)  
torch.manual_seed(41)

print("... Starting LeNet-5 Simulation ...")
start_time = time.time()
for i in range(N):
    xi = torch.sort(torch.rand(C))[0]
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)
    w = torch.rand(M_LeNet5) * (b - a) + a
    x_opt, lambda_opt, phi_opt, x_plus = knapsack_specialized_pruning_complete_parallel(xi, v, w, C)
training_time_LeNet5 = time.time() - start_time
print(f'Time spent for the LeNet-5 Simulation: {training_time_LeNet5:.2f} seconds')

print("... Starting LeNet-100 Simulation ...")
start_time = time.time()
for i in range(N):
    xi = torch.sort(torch.rand(C))[0]
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)
    w = torch.rand(M_LeNet100) * (b - a) + a
    x_opt, lambda_opt, phi_opt, x_plus = knapsack_specialized_pruning_complete_parallel(xi, v, w, C)
training_time_LeNet100 = time.time() - start_time
print(f'Time spent for the LeNet-100 Simulation: {training_time_LeNet100:.2f} seconds')

print(f"M_LeNet5 / M_LeNet100 = {M_LeNet5 / M_LeNet100}, "
      f"training_time_LeNet5 / training_time_LeNet100 {training_time_LeNet5 / training_time_LeNet100}")
print("-"*60)

# ------------------------------------------------------------------------------------------------------

print("... Starting LeNet-300 Simulation ...")
start_time = time.time()
for i in range(N):
    xi = torch.sort(torch.rand(C))[0]
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)
    w = torch.rand(M_LeNet300) * (b - a) + a
    x_opt, lambda_opt, phi_opt, x_plus = knapsack_specialized_pruning_complete_parallel(xi, v, w, C)
training_time_LeNet300 = time.time() - start_time
print(f'Time spent for the LeNet-300 Simulation: {training_time_LeNet300:.2f} seconds')

print(f"M_LeNet5 / M_LeNet300 = {M_LeNet5 / M_LeNet300}, "
      f"training_time_LeNet5 / training_time_LeNet300: {round(training_time_LeNet5 / training_time_LeNet300, 2)}")
print("-"*60)

# ------------------------------------------------------------------------------------------------------

print("... Starting AlexNet Simulation ...")
start_time = time.time()
for i in range(N):
    xi = torch.sort(torch.rand(C))[0]
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)
    w = torch.rand(M_AlexNet) * (b - a) + a
    x_opt, lambda_opt, phi_opt, x_plus = knapsack_specialized_pruning_complete_parallel(xi, v, w, C)
training_time_AlexNet = time.time() - start_time
print(f'Time spent for the AlexNet Simulation: {training_time_AlexNet:.2f} seconds')

print(f"M_LeNet5 / M_AlexNet = {M_LeNet5 / M_AlexNet}, "
      f"training_time_LeNet5 / training_time_AlexNet: {round(training_time_LeNet5 / training_time_AlexNet, 2)}")
print("-"*60)

# ------------------------------------------------------------------------------------------------------

print("... Starting VGG16 Simulation ...")
start_time = time.time()
for i in range(N):
    xi = torch.sort(torch.rand(C))[0]
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)
    w = torch.rand(M_VGG16) * (b - a) + a
    x_opt, lambda_opt, phi_opt, x_plus = knapsack_specialized_pruning_complete_parallel(xi, v, w, C)
training_time_VGG16 = time.time() - start_time
print(f'Time spent for the VGG16 Simulation: {training_time_VGG16:.2f} seconds')

print(f"M_LeNet5 / M_VGG16 = {M_LeNet5 / M_VGG16}, "
      f"training_time_LeNet5 / training_time_VGG16: {round(training_time_LeNet5 / training_time_VGG16, 2)}")
print("-"*60)

# ------------------------------------------------------------------------------------------------------

print("\n\nSimulation Completed.")