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
        self.W = None
        # create the parameter b and initialize it to zeros
        self.b = None
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
        out = None

        if not self.eval_mode:
            # You are in training mode.
            # Compute self.h_W = d(out)/d(W) and self.h_b = d(out)/d(b).
            # Remember to preserve the batch dimension as it is
            # collapsed only during the final gradient computation
            if self.h_W is None:
                pass

        return out

    def backward(self, grad):
        # grad is of shape n x d_out
        n = grad.shape[0]
        # Create placeholders for the gradients of W and b
        grad_W = None
        grad_b = None

        # Calculate the gradients for W and b. Use a for loop in the
        # beginning to ensure the correctness of your implementation
        pass
        
        # Average the gradients over the batch dimension
        grad_W = grad_W.mean(0)
        grad_b = grad_b.mean(0)

        # Update the parameters using the gradients
        pass

        # Return the grad for the previous layer
        grad_for_next = None

        return grad_for_next

class ReLU(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.sign = None

    def forward(self, X):
        if not self.eval_mode:
            # store the information required for the backward pass
            pass
        
        # Compute the ReLU activation
        out = None
        return out

    def backward(self, grad):
        # There is no gradient for ReLU since there are no parameters.
        # However, you must compute the gradient for the previous layer
        grad_for_next = None

        return grad_for_next

class MSE(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.grad_return = None

    def forward(self, yhat, y):
        if not self.eval_mode:
            # store the parts required for the backward pass
            pass
        
        # Calculate the mean squared error
        error = None
        return error

    def backward(self, grad=None):
        # There is no gradient for MSE since there are no parameters.
        # Return the gradient for the previous layer
        
        grad_for_next = None
        
        return grad_for_next