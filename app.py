from flask import Flask, request, render_template
from typing import List, Any
from data import SYMPTOM_MAPPING, DATASET
from ml_model import generate_recommendations

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Retrieve selected symptoms from the form submission
    selected_symptoms: List[str] = request.form.getlist('symptoms')

    # Generate exercise recommendations based on selected symptoms
    recommended_exercises = generate_recommendations(selected_symptoms)

    return render_template('index.html', recommendations=recommended_exercises)


if __name__ == '__main__':
    app.run(debug=True)
