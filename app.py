from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("teen_mental_health_model.pkl")

# Load Encoders
le_gender, le_platform, le_social = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])

    gender = le_gender.transform(
        [request.form['gender']]
    )[0]

    social_hours = float(
        request.form['daily_social_media_hours']
    )

    platform = le_platform.transform(
        [request.form['platform_usage']]
    )[0]

    sleep_hours = float(
        request.form['sleep_hours']
    )

    screen_time = float(
        request.form['screen_time_before_sleep']
    )

    academic = float(
        request.form['academic_performance']
    )

    physical = float(
        request.form['physical_activity']
    )

    social_level = le_social.transform(
        [request.form['social_interaction_level']]
    )[0]

    stress = int(request.form['stress_level'])

    anxiety = int(request.form['anxiety_level'])

    addiction = int(request.form['addiction_level'])

    data = np.array([[
        age,
        gender,
        social_hours,
        platform,
        sleep_hours,
        screen_time,
        academic,
        physical,
        social_level,
        stress,
        anxiety,
        addiction
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Depression Risk Detected"
    else:
        result = "No Depression Risk"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)