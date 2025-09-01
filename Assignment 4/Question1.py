import torch
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_rolls_range = range(2, 51)
num_experiments = [100, 500, 1000, 5000, 10000, 50000, 100000]

output_dir = "die_roll_histograms"
os.makedirs(output_dir, exist_ok=True)

results = []

for num_rolls in num_rolls_range:
    for n in num_experiments:
        # GPU accelerated random sampling
        rolls = torch.randint(1, 7, (n, num_rolls), device=device)
        sums = rolls.sum(dim=1)

        # Convert to CPU numpy for plotting
        sums_np = sums.cpu().numpy()

        mean_val = sums.float().mean().item()
        var_val = sums.float().var(unbiased=False).item()  # same as cupy.var
        results.append([num_rolls, n, mean_val, var_val])

        print(f"Rolls={num_rolls}, Experiments={n} → Mean={mean_val:.2f}, Var={var_val:.2f}")

        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(sums_np, bins=30, color="skyblue", edgecolor="black", density=True)
        plt.title(f"Histogram of {num_rolls} Dice Rolls (n={n})")
        plt.xlabel("Sum of rolls")
        plt.ylabel("Frequency (normalized)")

        filename = f"{output_dir}/hist_rolls{num_rolls}_exp{n}.png"
        plt.savefig(filename)
        plt.close()
