from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Training data
model = joblib.load('model.joblib')

@app.route("/predict")
def predict():
  w = float(request.args.get("w", 0))
  x = float(request.args.get("x", 0))
  
  y_pred = model.predict([[w, x]])[0]

  # Log prediction
  with open("output.txt", "w") as f:
    f.write(f"Input: w={w}, x={x}\nPrediction: {y_pred}\n")
    
  return jsonify({"w": w, "x": x, "prediction": y_pred})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
