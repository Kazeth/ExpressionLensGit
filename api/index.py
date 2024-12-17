from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask on Vercel is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Import predict logic if needed
    from predict import run_prediction
    data = request.json
    result = run_prediction(data)
    return jsonify(result)

# Vercel-specific handler
def handler(event, context):
    return app(event, context)
