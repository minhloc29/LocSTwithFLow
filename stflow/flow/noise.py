import torch


def zinb_prior(shape, total_count, logits, zi_logits):
    total_count = torch.as_tensor(total_count)
    logits = torch.as_tensor(logits)
    zi_logits = torch.as_tensor(zi_logits)

    nb_dist = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)
    nb_samples = nb_dist.sample(shape).squeeze(-1).float() # if the last dimension is 1, remove them

    zi_dist = torch.distributions.Bernoulli(logits=zi_logits)
    zi_mask = zi_dist.sample(shape).squeeze(-1).bool()

    return torch.where(zi_mask, torch.zeros_like(nb_samples), nb_samples)


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            self.prior_sampler = lambda shape: zinb_prior(
                shape,
                total_count=kwargs.get("total_count", None),
                logits=kwargs.get("logits", None),
                zi_logits=kwargs.get("zi_logits", None),
            )
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)


def gaussian_prior(shape):
    return torch.randn(shape)


def all_zeros(shape):
    return torch.zeros(shape)
