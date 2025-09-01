import torch
import matplotlib.pyplot as plt
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Observed data
n = 100   # number of reviews
k = 97    # positive reviews observed

# Different candidate success probabilities
success_rates = [0.90, 0.95, 0.97, 0.99]

output_dir = "./Assignment 4/binomial_histograms"
os.makedirs(output_dir, exist_ok=True)

for p in success_rates:
    trials = torch.full((100000,), n, device=device, dtype=torch.float32)
    probs = torch.full((100000,), p, device=device, dtype=torch.float32)

    samples = torch.binomial(trials, probs)
    
    samples_np = samples.cpu().numpy()

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(samples_np, bins=range(n+2), color="skyblue", edgecolor="black", density=True)
    plt.axvline(k, color="red", linestyle="dashed", linewidth=2, label=f"Observed = {k}")
    plt.title(f"Binomial Distribution (n={n}, p={p})")
    plt.xlabel("Number of Positive Reviews")
    plt.ylabel("Probability (normalized)")
    plt.legend()

    filename = f"{output_dir}/binomial_p{str(p).replace('.','_')}.png"
    plt.savefig(filename)
    plt.close()

print("Histograms saved in:", output_dir)
