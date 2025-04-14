from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
plt.switch_backend("agg") # this is to avoid a Matplotlib issue.

class States(Dataset):
    def __init__(self, num_steps=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        states_data = torch.load('code/states.pt') # Load your data here
        self.data = states_data['data'] # Load your actual 2D data here
        self.labels = states_data['labels'] # Load your labels here

        n_points = self.data.shape[0]
        self.n_points = n_points
        self.num_steps = num_steps

        self.steps = torch.linspace(start=-1.0, end=1.0, steps=self.num_steps) # Create the steps using linspace between -1 and 1
        self.beta = torch.linspace(start=10e-4, end=0.02, steps=self.num_steps) # Create beta according to the schedule in PDF
        self.alpha = 1.0 - self.beta # Compute alpha from beta
        self.alpha_bar = torch.stack([torch.prod(self.alpha[:t]) for t in range(len(self.alpha))]) # Compute alpha_bar from alpha

        self.mix_data()
    
    def refresh_eps(self):
        self.eps = None # Get a fresh set of epsilons

    def mix_data(self):
        self.all_data = []
        self.all_labels = []
        self.all_times = []
        self.refresh_eps()
        self.eps = self.eps.to("cpu")
        total_samples = len(self)

        for i in range(total_samples):
            data_idx = i % self.n_points
            step = i // self.n_points
            x = self.data[data_idx]
            t = self.steps[step]
            e = self.eps[i]
            x_ = self.calculate_noisy_data(x, t, e) # Create the noisy data from x, t, and e
            if self.labels is None:
                y = 0
            else:
                y = self.labels[data_idx]

            self.all_data.append(x_)
            self.all_times.append(t)
            self.all_labels.append(y)

        self.all_data = torch.stack(self.all_data).to(self.device)
        self.all_labels = torch.tensor(self.all_labels).to(self.device)
        self.all_steps = torch.tensor(self.all_times).to(self.device)
        self.eps = self.eps.to(self.device)

    def __len__(self):
        return self.n_points * self.num_steps

    def __getitem__(self, idx):
        data_idx = idx % self.n_points
        step = idx // self.n_points
        x = self.data[data_idx]
        t = self.steps[step]
        eps = torch.randn_like(x)
        x_ = self.calculate_noisy_data(x, t, eps) # create the noisy data from x, t, and eps
        if self.labels is None:
            y = 0
        else:
            y = self.labels[data_idx]
        return x_, t, eps, x, y

    def show(self, samples=None, save_to=None):
        if samples is None:
            samples = self.data
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.axis('equal')
        if save_to is not None:
            plt.savefig(save_to)
        plt.close()
        plt.clf()

    def calc_nll(self, generated):
        data_ = self.data.numpy()
        kde = gaussian_kde(data_.T)
        nll = -kde.logpdf(generated.T)
        return nll.mean()

    def calculate_noisy_data(self, x, t, e):
        return torch.sqrt(self.alpha_bar[t]) * x + torch.sqrt(1 - self.alpha_bar[t]) * e
