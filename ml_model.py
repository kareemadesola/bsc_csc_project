from typing import Dict, Union, List, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data import SYMPTOM_MAPPING, DATASET


class MLModel:
    def __init__(self):
        # Initialize the machine learning model
        self.model: Optional[RandomForestClassifier] = None

    def train(self, dataset: Dict[str, Union[List[List[float]], List[Any]]]):
        # Implement the training logic for the machine learning model using the provided dataset
        # Train the model to generate exercise recommendations based on symptoms

        # Split the dataset into features and target variables
        features = dataset['features']
        targets = dataset['targets']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        # Initialize and train a logistic regression model
        # rf = RandomForestClassifier(random_state=3)
        # rf.fit(X_train, y_train)
        # rf_pred = rf.predict(X_test)
        self.model = RandomForestClassifier(random_state=3)
        self.model.fit(X_train, np.array(y_train).flatten())

    def predict_exercises(self, analyzed_symptoms: List[List[float]]) -> List[Any]:
        # Implement the prediction logic for the machine learning model
        # Use the trained model to generate exercise recommendations based on analyzed symptoms

        # Make predictions using the trained model
        probabilities = self.model.predict_proba(analyzed_symptoms)
        predicted_indices = [np.argmax(probability) for probability in probabilities]
        return predicted_indices


def train_model(dataset: Dict[str, Union[List[List[float]], List[Any]]]) -> MLModel:
    # Train the machine learning model using the dataset
    model = MLModel()
    model.train(dataset)
    return model


def analyze_symptoms(symptoms: List[str]) -> List[float]:
    # Implement the logic to analyze the provided symptoms
    # Process and transform the symptoms data for input to the machine learning model

    # Get the list of symptoms from the user input
    # (This code is not used in the current version of the application, but it is included here for reference.)
    # user_symptoms = request.form.getlist('symptoms')

    # Initialize the processed symptoms list
    processed_symptoms = [0.0, 0.0, 0.0]

    # Convert each symptom to numerical representation or feature vector
    for symptom in symptoms:
        # Get the numerical value for the symptom
        numerical_value = SYMPTOM_MAPPING.get(symptom, 0.0)

        # Update the processed_symptoms list with the numerical value for the symptom
        for i in range(3):
            processed_symptoms[i] += numerical_value

    return processed_symptoms


def generate_recommendations(symptoms: List[str]) -> List[str]:
    # Analyze symptoms and generate exercise recommendations using the trained model
    analyzed_symptoms = analyze_symptoms(symptoms)

    model = train_model(DATASET)

    # Flatten the analyzed_symptoms list and reshape it to a 2D array with one row and three columns
    analyzed_symptoms = np.array(analyzed_symptoms).flatten().reshape(1, -1)

    # Print the shapes of the training features and analyzed symptoms
    features_shape = np.array(DATASET['features']).shape
    analyzed_shape = analyzed_symptoms.shape
    print("Features shape:", features_shape)
    print("Analyzed symptoms shape:", analyzed_shape)

    recommended_exercise_indices = model.predict_exercises(analyzed_symptoms)

    # Retrieve the recommended exercises from the DATASET using the indices
    recommended_exercises = [DATASET['targets'][index] for index in recommended_exercise_indices]
    return recommended_exercises


if __name__ == '__main__':
    # Test the model by generating exercise recommendations
    sample_symptoms = list(SYMPTOM_MAPPING.keys())
    recommendations = generate_recommendations(sample_symptoms)
    print(recommendations)
