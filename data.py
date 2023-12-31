SYMPTOM_MAPPING = {
    "pain": 0.9,
    "stiffness": 0.7,
    "swelling": 0.8,
    "fatigue": 0.6,
    "weakness": 0.5,
    "limited range of motion": 0.8,
    "joint redness": 0.7,
    "joint warmth": 0.7,
    "joint tenderness": 0.8,
    "morning stiffness": 0.6,
    "loss of appetite": 0.4,
    "weight loss": 0.5,
    "fever": 0.4,
    "muscle aches": 0.5,
    "decreased grip strength": 0.7,
    # Add more symptoms and their corresponding numerical values as needed
}

SYMPTOMS_OPTIONS = list(SYMPTOM_MAPPING.keys())
DATASET = {
    'features': [
        [0.9, 0.4, 0.5],
        [0.7, 0.6, 0.8],
        [0.8, 0.5, 0.7],
        [0.6, 0.4, 0.4],
        [0.5, 0.3, 0.8],
        [0.9, 0.7, 0.7],
        [0.7, 0.7, 0.8],
        [0.9, 0.5, 0.6],
        [0.6, 0.4, 0.4],
        [0.4, 0.5, 0.7],
        [0.7, 0.6, 0.9],
        [0.8, 0.7, 0.7],
        [0.9, 0.4, 0.8],
        [0.6, 0.5, 0.5],
        [0.5, 0.3, 0.7],
        [0.9, 0.8, 0.6],
        [0.7, 0.7, 0.8],
        [0.9, 0.5, 0.6],
        [0.6, 0.4, 0.4],
        [0.4, 0.5, 0.7],
        [0.7, 0.6, 0.9],
        [0.8, 0.7, 0.7],
        [0.9, 0.4, 0.8],
        [0.6, 0.5, 0.5],
        [0.5, 0.3, 0.7],
        [0.9, 0.8, 0.6],
        [0.7, 0.7, 0.8],
        [0.9, 0.5, 0.6],
        [0.6, 0.4, 0.4],
        [0.4, 0.5, 0.7],
    ],
    'targets': [
        'Leg Press',
        'Arm Curls',
        'Shoulder Press',
        'Walking',
        'Cycling',
        'Swimming',
        'Leg Raises',
        'Bicep Curls',
        'Triceps Dips',
        'Hip Abduction',
        'Leg Press',
        'Arm Curls',
        'Shoulder Press',
        'Walking',
        'Cycling',
        'Swimming',
        'Leg Raises',
        'Bicep Curls',
        'Triceps Dips',
        'Hip Abduction',
        'Leg Press',
        'Arm Curls',
        'Shoulder Press',
        'Walking',
        'Cycling',
        'Swimming',
        'Leg Raises',
        'Bicep Curls',
        'Triceps Dips',
        'Hip Abduction',
    ]
}
