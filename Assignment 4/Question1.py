import cupy as cp
import matplotlib.pyplot as plt
import os

num_rolls_range = range(2, 51)
num_experiments = [100, 500, 1000, 5000, 10000, 50000, 100000]

output_dir = "die_roll_histograms"
os.makedirs(output_dir, exist_ok=True)

results = []

for num_rolls in num_rolls_range:
    for n in num_experiments:
        # GPU accelerated random sampling
        rolls = cp.random.randint(1, 7, size=(n, num_rolls))
        sums = cp.sum(rolls, axis=1)

        # Convert back to NumPy for plotting
        sums_np = cp.asnumpy(sums)

        mean_val = cp.mean(sums).item()
        var_val = cp.var(sums).item()
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
