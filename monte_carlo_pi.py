import numpy as np
import matplotlib.pyplot as plt
from math import pi


def estimate_pi(n_points=100_000, seed=42, make_plots=True):
    """
    Generate uniformly distributed points in [-1, 1] x [-1, 1].
    Points satisfying x^2 + y^2 <= 1 are inside the unit circle.
    Estimate pi using: pi_hat = 4 * (inside_count / total_count).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)

    inside = x**2 + y**2 <= 1
    inside_count = int(inside.sum())
    pi_hat = 4 * inside_count / n_points

    print(f"Number of points:       {n_points:,}")
    print(f"Points inside circle:   {inside_count:,}")
    print(f"Estimated pi:           {pi_hat:.8f}")
    print(f"True pi:                {pi:.8f}")
    print(f"Absolute error:         {abs(pi_hat - pi):.8f}")

    if make_plots:
        shown = min(n_points, 10_000)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(x[:shown][inside[:shown]], y[:shown][inside[:shown]],
                   s=5, alpha=0.55, label="Inside circle")
        ax.scatter(x[:shown][~inside[:shown]], y[:shown][~inside[:shown]],
                   s=5, alpha=0.45, label="Outside circle")
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title(f"Random points (showing {shown:,})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        # Estimate after each point, for the convergence plot.
        running_inside = np.cumsum(inside)
        counts = np.arange(1, n_points + 1)
        running_estimate = 4 * running_inside / counts

        ax = axes[1]
        ax.plot(counts, running_estimate, linewidth=1, label="Running estimate")
        ax.axhline(pi, linestyle="--", linewidth=2, label="True pi")
        ax.set_xscale("log")
        ax.set_xlabel("Number of points (log scale)")
        ax.set_ylabel("Estimate of pi")
        ax.set_title("How the estimate converges")
        ax.legend()
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.show()

    return pi_hat


def compare_sample_sizes(sample_sizes=(100, 1_000, 10_000, 100_000, 1_000_000),
                         seed=42):
    """Compare estimates at increasing sample sizes using one shared sample."""
    rng = np.random.default_rng(seed)
    max_n = max(sample_sizes)
    x = rng.uniform(-1, 1, max_n)
    y = rng.uniform(-1, 1, max_n)
    inside = x**2 + y**2 <= 1

    print("\nSample-size comparison")
    print("-" * 55)
    print(f"{'N':>12} {'Estimated pi':>16} {'Absolute error':>18}")
    print("-" * 55)

    estimates = []
    errors = []
    for n in sample_sizes:
        estimate = 4 * inside[:n].mean()
        error = abs(estimate - pi)
        estimates.append(estimate)
        errors.append(error)
        print(f"{n:>12,} {estimate:>16.8f} {error:>18.8f}")

    plt.figure(figsize=(8, 4.5))
    plt.loglog(sample_sizes, errors, marker="o")
    plt.xlabel("Number of points N (log scale)")
    plt.ylabel("Absolute estimation error (log scale)")
    plt.title("Error versus sample size")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    estimate_pi(n_points=100_000, seed=42, make_plots=True)
    compare_sample_sizes()
