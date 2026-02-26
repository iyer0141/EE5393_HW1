import numpy as np
from collections import defaultdict

# Propensity Functions
def get_propensities(state):
    x1, x2, x3 = state

    # R1: 2X1 + X2 -> 4X3   (k1 = 1)
    a1 = 0.5 * x1 * (x1 - 1) * x2

    # R2: X1 + 2X3 -> 3X2   (k2 = 2)
    # 2 * x1 * (x3 choose 2)
    a2 = x1 * x3 * (x3 - 1)

    # R3: X2 + X3 -> 2X1   (k3 = 3)
    a3 = 3 * x2 * x3

    return a1, a2, a3

# Exact Distribution Propagation
def exact_distribution(initial_state, steps):

    current_dist = {initial_state: 1.0}

    # Stoichiometric updates
    changes = [
        (-2, -1, 4),   # R1
        (-1,  3, -2),  # R2
        ( 2, -1, -1)   # R3
    ]

    for step in range(steps):
        next_dist = defaultdict(float)

        for state, prob in current_dist.items():
            a1, a2, a3 = get_propensities(state)
            a_total = a1 + a2 + a3

            # Absorbing state case
            if a_total == 0:
                next_dist[state] += prob
                continue

            # Normalized transition probabilities
            p_trans = [a1/a_total, a2/a_total, a3/a_total]

            for i, p in enumerate(p_trans):
                if p > 0:
                    new_state = tuple(
                        s + c for s, c in zip(state, changes[i])
                    )
                    next_dist[new_state] += prob * p

        current_dist = next_dist

    return current_dist

# Computing Mean and Variance
def compute_statistics(distribution):
    means = np.zeros(3)
    second_moments = np.zeros(3)

    for state, prob in distribution.items():
        for i in range(3):
            means[i] += state[i] * prob
            second_moments[i] += (state[i]**2) * prob

    variances = second_moments - means**2
    return means, variances

# Main Execution
S0 = (9, 8, 7)
steps = 7

final_dist = exact_distribution(S0, steps)
means, variances = compute_statistics(final_dist)

# Output Final Answers
print("Exact Distribution after 7 Reaction Firings:")
print(f"Number of unique states: {len(final_dist)}\n")

print("Final Expected Values and Variances:")
species = ['X1', 'X2', 'X3']

for i in range(3):
    print(f"{species[i]}:")
    print(f"  Mean     = {means[i]:.6f}")
    print(f"  Variance = {variances[i]:.6f}\n")