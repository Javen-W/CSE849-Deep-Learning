import time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
plt.switch_backend("agg") # this is to avoid a Matplotlib issue.
import psutil

class States(Dataset):
    def __init__(self, num_steps=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Initial memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        print("Loading data...")
        start_time = time.time()
        states_data = torch.load('code/states.pt', weights_only=True)
        self.data = states_data['data'].to(self.device)  # Shape: (5000, 2)
        self.labels = states_data['labels'].to(self.device) if states_data['labels'] is not None else None
        print(f"Data loaded: {self.data.shape}, took {time.time() - start_time:.2f}s")
        print(f"Memory after load: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        self.n_points = self.data.shape[0]  # 5000
        self.n_steps = num_steps  # 500

        print("Computing noise schedule...")
        start_time = time.time()
        self.steps = torch.linspace(start=-1.0, end=1.0, steps=self.n_steps, device=self.device) # Create the steps using linspace between -1 and 1
        self.beta = torch.linspace(start=1e-4, end=0.02, steps=self.n_steps, device=self.device) # Create beta according to the schedule in PDF
        self.alpha = 1.0 - self.beta # Compute alpha from beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0) # Compute alpha_bar from alpha
        print(f"Noise schedule computed, took {time.time() - start_time:.2f}s")
        print(f"Memory after schedule: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        # Pre-compute dataset
        self.mix_data()

    def __len__(self):
        return self.n_points * self.n_steps  # 2,500,000

    def __getitem__(self, idx):
        # Return precomputed noisy data
        return (self.all_data[idx],
                self.all_steps[idx],
                self.eps[idx],
                self.data[idx % self.n_points],
                self.all_labels[idx])

    def generate_sample(self, idx):
        data_idx = idx % self.n_points
        step_idx = idx // self.n_points
        x = self.data[data_idx]
        t = self.steps[step_idx]
        e = torch.randn_like(x) if self.eps is None else self.eps[idx]
        x_ = self.calculate_noisy_data(x, step_idx, e)
        y = 0 if self.labels is None else self.labels[data_idx]
        return x_, t, e, x, y

    def mix_data(self):
        print("Initializing dataset...")
        start_time = time.time()
        total_samples = self.n_points * self.n_steps
        print(f"Precomputing {total_samples} noisy samples...")

        # Preallocate tensors
        self.all_data = torch.empty(total_samples, self.data.shape[1], device=self.device)
        self.all_labels = torch.empty(total_samples, dtype=torch.long, device=self.device)
        self.all_steps = torch.empty(total_samples, device=self.device)
        self.eps = torch.randn(total_samples, self.data.shape[1], device=self.device) # Get a fresh set of noise

        for i in range(total_samples):
            if i % 100_000 == 0:
                print(f"Processing sample {i}/{total_samples}, "
                      f"memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB, "
                      f"GPU memory: {torch.cuda.memory_allocated(self.device) / 1024 ** 2:.2f} MB")

            # Generate & cache sample
            x_, t, e, x, y = self.generate_sample(i)
            self.all_data[i] = x_
            self.all_steps[i] = t
            self.all_labels[i] = y

            # Periodic cleanup
            if i % 100_000 == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        print(f"Dataset initialized, took {time.time() - start_time:.2f}s")
        print(f"Memory after mix_data: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

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
