from flask import Flask, request, render_template

from data import SYMPTOMS_OPTIONS
from ml_model import generate_recommendations

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve selected symptoms from the form submission
        selected_symptoms = request.form.getlist('symptoms')

        if not selected_symptoms:
            # If no symptoms are selected, return an empty list as recommendations
            return render_template('index.html', symptoms=SYMPTOMS_OPTIONS, recommendations=[])

        # Generate exercise recommendations based on selected symptoms
        recommended_exercises = generate_recommendations(selected_symptoms)

        return render_template('index.html', symptoms=SYMPTOMS_OPTIONS, recommendations=recommended_exercises)
    else:
        return render_template('index.html', symptoms=SYMPTOMS_OPTIONS)


if __name__ == '__main__':
    app.run(debug=True)
