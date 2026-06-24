from flask import Flask, render_template, request
import joblib
import pandas as pd
import json

app = Flask(__name__)
model = joblib.load("taxi_price_pipeline.joblib")
with open("selected_features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

TIME_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {}
    if request.method == "POST":
        try:
            values = {
                "Base_Fare": float(request.form.get("Base_Fare", "")),
                "Trip_Distance_km": float(request.form.get("Trip_Distance_km", "")),
                "Per_Km_Rate": float(request.form.get("Per_Km_Rate", "")),
                "Trip_Duration_Minutes": float(request.form.get("Trip_Duration_Minutes", "")),
                "Per_Minute_Rate": float(request.form.get("Per_Minute_Rate", "")),
                "Traffic_Conditions": request.form.get("Traffic_Conditions", "Medium")
            }
            if values["Trip_Distance_km"] < 0 or values["Trip_Duration_Minutes"] < 0:
                raise ValueError("La distancia y la duración no pueden ser negativas.")
            input_df = pd.DataFrame([values], columns=FEATURES)
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            error = "Revisa los datos ingresados. " + str(exc)
    return render_template("index.html", prediction=prediction, error=error, values=values, traffic_options=TRAFFIC_OPTIONS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
