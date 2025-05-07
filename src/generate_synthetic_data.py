import numpy as np
import pandas as pd

np.random.seed(42)

# Number of samples
n_samples = 1000

# Number of covariates
n_covariates = 10

# Generate random covariates (patient characteristics)
X = np.random.normal(size=(n_samples, n_covariates))

# Generate random binary treatment assignments (50% treated, 50% control)
treatment = np.random.binomial(1, 0.5, size=n_samples)

# Define true underlying functions for potential outcomes
def outcome_control(X):
    return 2 + X @ np.random.uniform(-1, 1, size=n_covariates)

def outcome_treated(X):
    return 4 + X @ np.random.uniform(-1, 1, size=n_covariates) + 0.5 * np.sin(X[:,0])

# Generate potential outcomes
y_control = outcome_control(X) + np.random.normal(0, 1, size=n_samples)
y_treated = outcome_treated(X) + np.random.normal(0, 1, size=n_samples)

# Realized observed outcomes based on treatment assignment
y_observed = treatment * y_treated + (1 - treatment) * y_control

# Assemble into DataFrame
df_synthetic = pd.DataFrame(X, columns=[f'covariate_{i}' for i in range(n_covariates)])
df_synthetic['treatment'] = treatment
df_synthetic['y_observed'] = y_observed
df_synthetic['y_control'] = y_control
df_synthetic['y_treated'] = y_treated

# Save dataset to CSV
df_synthetic.to_csv('data/Synthetic/synthetic_basket_trial.csv', index=False)
