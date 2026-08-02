import torch
from .noise import PriorSampler


class Interpolant:
    def __init__(self, prior_sample_type, normalize=True, device=None, **kwargs):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.prior_sampler = PriorSampler(prior_sample_type, device=self.device, **kwargs)
        self.normalize = normalize

    def sample_from_prior(self, shape, device): # just to get x0
        exp = self.prior_sampler.sample(shape).to(device)
        if self.normalize:
            exp = torch.log(exp + 1)
        return exp

    def sample_t(self, shape): # random t [0, 1]
        return torch.rand(shape)

    def corrupt_exp(self, exp): # get x_t = (1-t)x0 + tx1
        # exp: [B, n_cells, n_genes] -> normal ones
        # build everything on the same device as exp (honors --device cuda:1 etc.)
        t = self.sample_t((exp.shape[0],)).to(exp.device)
        if exp.shape[0] > 1:
            t = t.squeeze(-1)
        exp_0 = self.sample_from_prior(exp.shape, exp.device)
        return exp_0 * (1 - t[:, None, None]) + exp * t[:, None, None], t

    def denoise(self, exp_1, exp_t, t, d_t):
       
        exp_vf = (exp_1 - exp_t) / (1 - t[:, None, None])
        return exp_t + d_t[:, None, None] * exp_vf
