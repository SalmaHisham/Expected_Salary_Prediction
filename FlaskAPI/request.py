import requests 

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}

test_data = [0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 1.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 1.0,
 0.0,
 0.0,
 0.0,
 0.0,
 1.0,
 0.0,
 0.0,
 1.0]

data = {"input": test_data}
r = requests.get(URL,headers=headers, json=data) 
# r.json()