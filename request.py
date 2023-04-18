import requests
import json

url = 'http://localhost:5000/predict'
model = 'base1'
features = {
    'feature1': 2500.0,
    'feature2': 1.2,
    'feature3': 1.2,
    'feature4': 1.2,
    'feature5': 1.2,
    'feature6': 1.2,
    'feature7': 1.2,
    'feature8': 1.2,
    'feature9': 1.2,
    'feature10': 0.0,
    'feature11': 0.0,
    'feature12': 0.0,
    'feature13': 0.0,
    'feature14': 0.0,
    'feature15': 0.0,
    'feature16': 0.0,
    'feature17': 0.0,
    'feature18': 0.0,
    'feature19': 0.0,
    'feature20': 0.0,
    'feature21': 0.0,
    'feature22': 0.0,
    'feature23': 0.0,
    'feature24': 0.0,
    'feature25': 0.0,
    'feature26': 0.0,
    'feature27': 0.0,
    'feature28': 0.0,
    'feature29': 0.0,
    'feature30': 0.0,
    'feature31': 0.0,
    'feature32': 0.0,
    'feature33': 0.0,
    'feature34': 0.0,
    'feature35': 0.0,
    'feature36': 0.0,
    'feature37': 0.0,
    'feature38': 0.0,
    'feature39': 0.0,
    'feature40': 0.0,
    'feature41': 0.0,
    'feature42': 0.0,
    'feature43': 0.0,
    'feature44': 0.0,
    'feature45': 0.0,
    'feature46': 0.0,
    'feature47': 0.0,
    'feature48': 0.0,
    'feature49': 0.0,
    'feature50': 0.0,
    'feature51': 0.0,
    'feature52': 0.0,
    'feature53': 0.0,
    'feature54': 0.0,
}

data = {
    'model': model,
    'features': features
}

headers = {'Content-type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    prediction = response.json()['prediction']
    print(f'The predicted output is: {prediction}')
else:
    print('Error:', response.json()['error'])