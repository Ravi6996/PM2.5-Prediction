import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'year':1,'month':2, 'day':3,'hour':4 ,'temperature':5 ,'pressure':6 ,'rain':7 ,'wind_direction':'N' ,'wind_speed':9})

print(r.json())