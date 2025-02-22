import torch

# This is your BaseUnit class. You will inherit from this class to
# create custom layers.
class BaseUnit:
    def __init__(self, lr):
        self.eval_mode = False
        self.lr = lr

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(BaseUnit):
    def __init__(self, d_in, d_out, lr=1e-3):
        super().__init__(lr)
        # create the parameter W and initialize it from a normal
        # distribution with mean 0 and std 0.05. Check torch.randn
        # for this.
        self.W = torch.randn(size=(d_in, d_out)) * 0.05
        # create the parameter b and initialize it to zeros
        self.b = torch.zeros(size=(d_out, ))
        self.d_in = d_in
        self.d_out = d_out
        # self.grad_comps for each parameter
        self.h_W = None
        self.h_b = None

    def forward(self, X):
        # X is a batch of data of shape n x d_in
        n = X.shape[0]
        # calculate out = X @ W + b. Remember to reshape b so that it
        # adds elementwise to each row.
        out = X @ self.W + self.b.reshape(1, -1)

        if not self.eval_mode:
            # You are in training mode.
            # Compute self.h_W = d(out)/d(W) and self.h_b = d(out)/d(b).
            # Remember to preserve the batch dimension as it is
            # collapsed only during the final gradient computation
            self.h_W = X
            self.h_b = torch.ones(n, 1)

        return out

    def backward(self, grad):
        # grad is of shape n x d_out
        n = grad.shape[0]
        # Create placeholders for the gradients of W and b
        grad_W = torch.zeros(self.d_in, self.d_out)
        grad_b = torch.zeros(self.d_out)

        # Calculate the gradients for W and b. Use a for loop in the
        # beginning to ensure the correctness of your implementation
        for i in range(n):
            grad_W += torch.outer(self.h_W[i], grad[i])
            grad_b += grad.sum(dim=0)
        
        # Average the gradients over the batch dimension
        grad_W = grad_W.mean(dim=0)
        grad_b = grad_b.mean(dim=0)

        # Return the grad for the previous layer
        grad_for_next = grad @ self.W.T

        # Update the parameters using the gradients
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

        return grad_for_next

class ReLU(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.sign = None

    def forward(self, X):
        if not self.eval_mode:
            # store the information required for the backward pass
            self.sign = (X > 0).float()
        
        # Compute the ReLU activation
        out = torch.max(X, torch.zeros_like(X))
        return out

    def backward(self, grad):
        # There is no gradient for ReLU since there are no parameters.
        # However, you must compute the gradient for the previous layer
        # d(ReLU)/dx = 1 if x > 0, 0 otherwise
        grad_for_next = grad * self.sign

        return grad_for_next

class MSE(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.grad_return = None

    def forward(self, yhat, y):
        n = yhat.shape[0]  # batch size
        if not self.eval_mode:
            # store the parts required for the backward pass
            # d(MSE)/d(yhat) = (2/n) * (yhat - y)
            self.grad_return = (2 / n) * (yhat - y)
        
        # Calculate the mean squared error
        error = torch.mean((yhat - y) ** 2) * (1 / n)
        return error

    def backward(self, grad=None):
        # There is no gradient for MSE since there are no parameters.
        # Return the gradient for the previous layer
        grad_for_next = self.grad_return
        return grad_for_next