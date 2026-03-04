
import numpy as np

class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, x: np.ndarray):
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        self.sigma = np.nanstd(x, axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x: np.ndarray):
        return x * self.sigma + self.mu

    def state_dict(self):
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()} # type: ignore

    def load_state_dict(self, d):
        self.mu = np.array(d["mu"]) if not isinstance(d["mu"], np.ndarray) else d["mu"]
        self.sigma = np.array(d["sigma"]) if not isinstance(d["sigma"], np.ndarray) else d["sigma"]