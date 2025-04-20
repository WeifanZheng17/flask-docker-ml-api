from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ────────────────
# QUESTION 1: TRAIN & DISPLAY REGRESSION
# ────────────────

# 1) Build the DataFrame
data = {
    'W': [0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1],
    'X': [19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,
          20.2,27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8],
    'Y': [137,118,124,124,120,129,122,142,128,114,
          132,130,130,112,132,117,134,132,121,128]
}
df = pd.DataFrame(data)

# 2) Specify regressors + intercept
X = sm.add_constant(df[['W','X']])  # columns: const, W, X
y = df['Y']

# 3) Fit OLS
ols_res = sm.OLS(y, X).fit()

# 4) Print summary to console
print("\n=========== OLS SUMMARY ===========")
print(ols_res.summary())
with open("q1_summary.txt", "w") as f:
    f.write(ols_res.summary().as_text())

# 5) Extract ATE and p‑value
tau_hat = ols_res.params['W']
pval_tau = ols_res.pvalues['W']
with open("q1_results.txt", "w") as f:
    f.write(f"ATE (τ̂) = {tau_hat:.4f}\n")
    f.write(f"p‑value = {pval_tau:.4f}\n")

print(f"\n>>> ATE (τ̂) = {tau_hat:.4f}, p‑value = {pval_tau:.4f}\n")


# ────────────────
# QUESTION 2: FLASK API
# ────────────────

app = Flask(__name__)

@app.route("/predict")
def predict():
  w = float(request.args.get("w", 0))
  x = float(request.args.get("x", 0))
  
  # build new data for prediction
  new_X = sm.add_constant(pd.DataFrame({'W': [w], 'X': [x]}))
  
  y_pred = ols_res.predict(new_X)[0]

  # Log prediction
  with open("output.txt", "w") as f:
    f.write(f"Input: w={w}, x={x}\nPrediction: {y_pred}\n")
    
  return jsonify({"w": w, "x": x, "prediction": y_pred})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
