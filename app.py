from flask import Flask, request, render_template, redirect, session

from data import SYMPTOMS_OPTIONS
from ml_model import generate_recommendations

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong, random secret key

# Define a variable to track the logged-in state
logged_in = False


@app.route('/login', methods=['GET', 'POST'])
def login():
    global logged_in
    if request.method == 'POST':
        # Check username and password (you can replace this with your authentication logic)
        username = request.form['username']
        password = request.form['password']

        # Replace this with your actual authentication logic
        if username == 'adesola' and password == 'kareem':
            # Successful login
            logged_in = True  # Update the logged-in state
            session['username'] = username  # Store the username in the session
            return redirect('/')
        else:
            # Login failed, display an error message
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    global logged_in
    if not logged_in:
        return redirect('/login')  # Redirect to the login page if not logged in

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
