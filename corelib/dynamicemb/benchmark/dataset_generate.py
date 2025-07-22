
import torch
import numpy as np
def zipf(min_val, max_val, exponent, size, device):
    """
    Generates Zipf-like random variables in the inclusive range [min_val, max_val).

    Args:
        min_val (int): Minimum value (inclusive, must be ≥0).
        max_val (int): Maximum value (exclusive).
        exponent (float): Exponent parameter (a > 0).
        size (int): Output shape.

    Returns:
        torch.Tensor: Sampled values of specified size.
    """

    # Generate integer values and probabilities
    values = torch.arange(min_val + 1, max_val + 1, dtype=torch.long, device=device)
    probs = 1.0 / (values.float() ** exponent)
    probs_normalized = probs / probs.sum()
    print(f"Exponent:{exponent}")
    for x in probs_normalized:
      print(x.item())

    k = torch.arange(min_val, max_val, dtype=torch.long, device=device)
    perm = torch.randperm(k.size(0), device=device)
    k_shuffled = k[perm]
    print(f"Permute:")
    for x in k_shuffled:
      print(x.item()) 

    probs_np = probs_normalized.cpu().numpy()
    samples = np.random.choice(k_shuffled.cpu().numpy(), size=size, replace=True, p=probs_np)
    samples = torch.tensor(samples, device=probs_normalized.device)
    print(f"Result:")
    for x in samples:
      print(x.item())    
    

if __name__ == "__main__":
  zipf(0, 100, 1.05, 100, torch.device("cuda:0"))
  zipf(0, 100, 1.2, 100, torch.device("cuda:0"))