SYMPTOM_MAPPING = {
    "pain": 0.8,
    "stiffness": 0.6,
    "swelling": 0.7,
    "fatigue": 0.5,
    "weakness": 0.4,
    "limited range of motion": 0.7,
    "joint redness": 0.6,
    "joint warmth": 0.6,
    "joint tenderness": 0.7,
    "morning stiffness": 0.5,
    "loss of appetite": 0.3,
    "weight loss": 0.4,
    "fever": 0.3,
    "muscle aches": 0.4,
    "decreased grip strength": 0.6,
    # Add more symptoms and their corresponding numerical values as needed
}

SYMPTOMS_OPTIONS = list(SYMPTOM_MAPPING.keys())

DATASET = {
    'features': [
        [0.8], [0.6], [0.7], [0.5], [0.4], [0.7], [0.6], [0.6], [0.7], [0.5],
        [0.3], [0.4], [0.3], [0.4], [0.6],
        # Add more feature values as needed
    ],
    'targets': [
        'Exercise A', 'Exercise B', 'Exercise C', 'Exercise D', 'Exercise E',
        'Exercise F', 'Exercise G', 'Exercise H', 'Exercise I', 'Exercise J',
        'Exercise K', 'Exercise L', 'Exercise M', 'Exercise N', 'Exercise O',
        # Add more target values as needed
    ]
}
