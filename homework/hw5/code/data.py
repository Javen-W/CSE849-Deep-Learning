import time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
plt.switch_backend("agg") # this is to avoid a Matplotlib issue.

class States(Dataset):
    def __init__(self, num_steps=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        states_data = torch.load('code/states.pt', weights_only=True) # Load your data here
        self.data = states_data['data'].to(self.device) # Load your actual 2D data here (5000x2)
        self.labels = states_data['labels'] # Load your labels here

        self.n_points = self.data.shape[0]
        self.n_steps = num_steps

        self.steps = torch.linspace(start=-1.0, end=1.0, steps=self.n_steps, device=self.device) # Create the steps using linspace between -1 and 1
        self.beta = torch.linspace(start=10e-4, end=0.02, steps=self.n_steps, device=self.device) # Create beta according to the schedule in PDF
        self.alpha = 1.0 - self.beta # Compute alpha from beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0) # Compute alpha_bar from alpha

        self.mix_data()

    def mix_data(self):
        start_time = time.time()

        # Preallocate tensor for efficiency
        total_samples = len(self)
        self.all_data = torch.empty(total_samples, self.data.shape[1], device=self.device)
        self.all_labels = torch.empty(total_samples, dtype=torch.long, device=self.device)
        self.all_steps = torch.empty(total_samples, device=self.device)
        self.eps = torch.randn(total_samples, self.data.shape[1], device=self.device)  # Get a fresh set of noise

        for i in range(total_samples):
            data_idx = i % self.n_points
            step = i // self.n_points
            x = self.data[data_idx]
            t = self.steps[step]
            e = self.eps[i]
            x_ = self.calculate_noisy_data(x, step, e)  # Use step directly
            y = 0 if self.labels is None else self.labels[data_idx]

            self.all_data[i] = x_
            self.all_steps[i] = t
            self.all_labels[i] = y

        print(f"mix_data(): {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return self.n_points * self.n_steps  # 2,500,000

    def __getitem__(self, idx):
        # Return precomputed noisy data
        return (self.all_data[idx],
                self.all_steps[idx],
                self.eps[idx],
                # torch.randn_like(self.all_data[idx]),  # Fresh eps for training
                self.data[idx % self.n_points],
                self.all_labels[idx])

    def show(self, samples=None, save_to=None):
        if samples is None:
            samples = self.data
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.axis('equal')
        if save_to is not None:
            plt.savefig(save_to)
        plt.close()
        plt.clf()

    def calc_nll(self, generated):
        data_ = self.data.cpu().numpy()
        kde = gaussian_kde(data_.T)
        nll = -kde.logpdf(generated.T)
        return nll.mean()

    def calculate_noisy_data(self, x, step_idx, e):
        alpha_bar_t = self.alpha_bar[step_idx]
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e
