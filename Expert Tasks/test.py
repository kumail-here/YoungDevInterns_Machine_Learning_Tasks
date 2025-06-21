# test.py

import requests

url = 'http://127.0.0.1:5000/predict'
sample_data = {
    'features': [5.1, 3.5, 1.4, 0.2]  # Example Iris input
}

response = requests.post(url, json=sample_data)
print("Predicted class:", response.json()['prediction'])
