import pandas as pd
import statsmodels.api as sm

# Build the DataFrame
data = {
    'W': [0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1],
    'X': [19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,
          20.2,27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8],
    'Y': [137,118,124,124,120,129,122,142,128,114,
          132,130,130,112,132,117,134,132,121,128]
}
df = pd.DataFrame(data)

# Specify regressors and add the intercept
X = sm.add_constant(df[['W','X']])  # columns: const, W, X
y = df['Y']

# Fit OLS and print summary, as the previous 'LinearRegression' object has no attribute 'summary'
model = sm.OLS(y, X).fit()
print(model.summary())

alpha_hat = model.params['const']
tau_hat   = model.params['W']
beta_hat  = model.params['X']

pval_tau  = model.pvalues['W']

print(f"α̂  = {alpha_hat:.4f}")
print(f"τ̂  = {tau_hat:.4f} (p‑value = {pval_tau:.4f})")
print(f"β̂  = {beta_hat:.4f}")
