"""
An example of a multi-objective optimization problem that could occur in a real-life scenario.
Imagine a company wants to minimize production costs while maximizing the quality of its products.

We'll represent this with two functions:
- one for cost and another for quality.
- The goal is to find the production parameters (like quantity of raw materials) that achieve these objectives.

We'll use the `scipy.optimize` module for this example.
"""

from scipy.optimize import minimize
import numpy as np


# Objective 1: Minimize production cost
# This function represents the cost of production, which depends on two variables:
# x[0] - Quantity of material A
# x[1] - Quantity of material B
# The cost is a function of these quantities, and we assume it increases linearly with them.
def production_cost(x):
    cost_A = 5  # Cost per unit of material A
    cost_B = 10  # Cost per unit of material B
    return cost_A * x[0] + cost_B * x[1]


# Objective 2: Maximize product quality
# This function represents the quality of the product, which also depends on the two variables.
# We assume the quality increases with more of material B but decreases if too much of material A is used.
def product_quality(x):
    return -np.sqrt(x[0]) + 2 * np.log1p(x[1])


# Multi-objective function
# This function combines both objectives. We need to remember that 'minimize' only minimizes,
# so to maximize product quality, we minimize the negative of the quality function.
def multi_objective(x):
    return production_cost(x) - product_quality(x)


def main():
    # Constraints
    # These represent limits on the quantities of materials A and B.
    # For instance, due to budget or supply limitations.
    cons = (
        {'type': 'ineq', 'fun': lambda x: 10 - x[0]},  # Limit for material A
        {'type': 'ineq', 'fun': lambda x: 5 - x[1]}  # Limit for material B
    )

    # Bounds
    # These represent the minimum and maximum values for each variable.
    # We assume that at least 1 unit and no more than 10 units of each material can be used.
    bounds = [(1, 10), (1, 10)]

    # Initial guess
    x0 = [1, 1]

    # Optimizer
    res = minimize(multi_objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res)
    print(f"Optimal solution: Material A = {res.x[0]}, Material B = {res.x[1]}")
    print(f"Minimum production cost: {production_cost(res.x)}")
    print(f"Maximum product quality: {product_quality(res.x)}")


if __name__ == '__main__':
    main()