import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def theta(x1, x2):
    # Calculate theta as provided in the question
    pass

def r(x1, x2):
    # Calculate r as provided in the question
    pass    

def y(x1, x2):
    # First calculate r and theta using the functions defined above and
    # then calculate y as provided in the question.
    pass

def grad_theta_x1(x1, x2):
    # Calculate the gradient of theta w.r.t. x1
    pass

def grad_theta_x2(x1, x2):
    # Calculate the gradient of theta w.r.t. x2
    pass

def grad_r_x1(x1, x2):
    # Calculate the gradient of r w.r.t. x1
    pass

def grad_r_x2(x1, x2):
    # Calculate the gradient of r w.r.t. x2
    pass

def grad_y_theta(rval, thetaval):
    # Calculate the gradient of y w.r.t. theta
    pass

def grad_y_r(rval, thetaval):
    # Calculate the gradient of y w.r.t. r
    pass

def grad_y_x1(x1, x2):
    # Calculate the gradient of y w.r.t. x1. First
    # calculate the gradients of y w.r.t. theta and r using the
    # functions defined above. Then calculate the gradients of theta and
    # r w.r.t. x1 using the functions defined above. Finally, use the
    # chain rule to calculate the gradient of y w.r.t. x1.
    pass

def grad_y_x2(x1, x2):
    # Calculate the gradient of y w.r.t. x2. First
    # calculate the gradients of y w.r.t. theta and r using the
    # functions defined above. Then calculate the gradients of theta and
    # r w.r.t. x2 using the functions defined above. Finally, use the
    # chain rule to calculate the gradient of y w.r.t. x2.
    pass



seeds = [1, 2, 3, 4, 5] # seeds
num_steps = 2000 # maximum number of steps
tol = 1e-3 # error tolerance
lams_list = [1e-4, 1e-3, 1e-2, 1e-1, 1] # step size list

# For plotting
x1 = torch.linspace(-5, 5, 100)
x2 = torch.linspace(-5, 5, 100)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

Y = y(X1, X2)
os.makedirs("plots", exist_ok=True)

for lam_idx, lam in enumerate(lams_list):
    for seed in seeds:
        print("Running: lam =", lam, "seed =", seed)
        torch.manual_seed(seed)

        # Find random starting points for x1 and x2 between [-5, 5] x
        # [-5, 5] using torch.rand
        x1 = None
        x2 = None

        # Store the values for plotting
        y_vals = []
        x1_vals = [x1.item()]
        x2_vals = [x2.item()]

        for step in range(num_steps):
            # Calculate the value of y
            yval = None

            # Store the value
            y_vals.append(yval)

            # Calculate the gradients
            x1_grad = grad_y_x1(x1, x2)
            x2_grad = grad_y_x2(x1, x2)

            # Write the update equation for x1 and x2

            # Store the updated values for plotting
            x1_vals.append(x1.item())
            x2_vals.append(x2.item())

            # If the error is less than tol, break
            if yval < tol:
                break
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        im = axs[0].contourf(X1, X2, Y, cmap="Spectral", levels=100)
        axs[0].plot(x1_vals, x2_vals, linewidth=2, marker=".", color="black", markersize=2)
        axs[0].set_xlabel("X1")
        axs[0].set_ylabel("X2")
        fig.colorbar(im, ax=axs[0])

        axs[1].plot(torch.arange(len(y_vals)), y_vals)
        axs[1].set_yscale("log")
        axs[1].set_xlabel("step")
        axs[1].grid(True)
        axs[1].set_ylabel("y")

        fig.suptitle(f"Step size: {lam}, seed: {seed}")

        fig.tight_layout()
        fig.savefig(f"plots/q1_{lam}_{seed}.png", dpi=300)
        plt.clf()
        plt.close(fig)
