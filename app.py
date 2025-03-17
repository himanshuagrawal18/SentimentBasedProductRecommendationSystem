from flask import Flask, request, render_template, jsonify
from model import product_recommendations_user

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    error_message = None

    if request.method == "POST":
        user = request.form["user"]
        recommendations = product_recommendations_user(user)

        if isinstance(recommendations, str):  # If function returns an error message
            error_message = recommendations
            recommendations = None
        elif recommendations.empty:  # If DataFrame is empty
            error_message = "No recommendations found."
            recommendations = None

    return render_template("index.html", recommendations=recommendations, error_message=error_message)

@app.route("/api/recommendations", methods=["GET"])
def api_recommendations():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "User parameter is required"}), 400

    recommendations = product_recommendations_user(user)

    if isinstance(recommendations, str):  # If an error message is returned
        return jsonify({"error": recommendations}), 400

    return jsonify(recommendations.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
