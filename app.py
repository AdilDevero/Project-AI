from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    features = np.array([float(x) for x in data.values()]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
