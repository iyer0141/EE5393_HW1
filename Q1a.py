import numpy as np

def get_propensities(state):
    x1, x2, x3 = state

    # R1: 2X1 + X2 -> 4X3
    a1 = 0.5 * x1 * (x1 - 1) * x2 if x1 >= 2 and x2 >= 1 else 0.0

    # R2: X1 + 2X3 -> 3X2
    a2 = x1 * x3 * (x3 - 1) if x1 >= 1 and x3 >= 2 else 0.0

    # R3: X2 + X3 -> 2X1
    a3 = 3 * x2 * x3 if x2 >= 1 and x3 >= 1 else 0.0

    return a1, a2, a3

def run_gillespie_trajectory(initial_state, max_steps=10000):
    state = np.array(initial_state, dtype=int)

    # Checking if starting state already meets conditions
    hit_c1 = state[0] >= 150
    hit_c2 = state[1] < 10
    hit_c3 = state[2] > 100

    for _ in range(max_steps):
        a1, a2, a3 = get_propensities(state)
        a_total = a1 + a2 + a3

        # If no reactions can fire, the system has halted
        if a_total == 0:
            break

        # Generating a random number to select the next reaction
        r = np.random.rand() * a_total

        # Applying stoichiometry based on the chosen reaction
        if r < a1:
            state += np.array([-2, -1, 4])  # R1 fires
        elif r < a1 + a2:
            state += np.array([-1, 3, -2])  # R2 fires
        else:
            state += np.array([2, -1, -1])  # R3 fires

        # Checking if conditions are met during this step
        if not hit_c1 and state[0] >= 150: hit_c1 = True
        if not hit_c2 and state[1] < 10:   hit_c2 = True
        if not hit_c3 and state[2] > 100:  hit_c3 = True

        # Stop early if all conditions are met
        if hit_c1 and hit_c2 and hit_c3:
            break

    return hit_c1, hit_c2, hit_c3

def estimate_probabilities(initial_state, n_simulations=200, max_steps=10000):

    print(f"Running {n_simulations} simulations starting from {initial_state}...\n")

    count_c1 = 0
    count_c2 = 0
    count_c3 = 0

    for _ in range(n_simulations):
        c1, c2, c3 = run_gillespie_trajectory(initial_state, max_steps)
        if c1: count_c1 += 1
        if c2: count_c2 += 1
        if c3: count_c3 += 1

    p_c1 = count_c1 / n_simulations
    p_c2 = count_c2 / n_simulations
    p_c3 = count_c3 / n_simulations

    return p_c1, p_c2, p_c3

if __name__ == "__main__":
    S0 = [110, 26, 55]

    # Running the Monte Carlo estimation with exactly 200 iterations
    p1, p2, p3 = estimate_probabilities(S0, n_simulations=200)

    print("--- Estimated Probabilities (200 Iterations) ---")
    print(f"Pr(C1: X1 >= 150) = {p1:.4f}  ({p1 * 100:.2f}%)")
    print(f"Pr(C2: X2 < 10)   = {p2:.4f}  ({p2 * 100:.2f}%)")
    print(f"Pr(C3: X3 > 100)  = {p3:.4f}  ({p3 * 100:.2f}%)")