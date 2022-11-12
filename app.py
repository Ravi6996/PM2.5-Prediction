import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Year = int(request.form['year'])
    Month = int(request.form['month'])
    Day = int(request.form['day'])
    Hour = int(request.form['hour'])
    Temperature = float(request.form['temperature'])
    Pressure = float(request.form['pressure'])
    Rain = float(request.form['rain'])
    
    Wind_direction = request.form['wind_direction']
    if(Wind_direction=='NW'):
        Wind_direction=0
    elif(Wind_direction=='NNW'):
         Wind_direction=1
    elif(Wind_direction=='WNW'):
         Wind_direction=2
    elif(Wind_direction=='N'):
         Wind_direction=3
    elif(Wind_direction=='NNE'):
         Wind_direction=4
    elif(Wind_direction=='WSW'):
         Wind_direction=5
    elif(Wind_direction=='W'):
         Wind_direction=6
    elif(Wind_direction=='SW'):
         Wind_direction=7
    elif(Wind_direction=='SSW'):
         Wind_direction=8      
    elif(Wind_direction=='S'):
         Wind_direction=9
    elif(Wind_direction=='NE'):
         Wind_direction=10
    elif(Wind_direction=='SSE'):
         Wind_direction=11
    elif(Wind_direction=='SE'):
         Wind_direction=12
    elif(Wind_direction=='E'):
         Wind_direction=13
    elif(Wind_direction=='ENE'):
         Wind_direction=14
    else:
         Wind_direction=15
    wind_speed=float(request.form['wind_speed'])
    final_features=[[Year,Month,Day, Hour, Temperature,Pressure,Rain,Wind_direction,wind_speed]]
    prediction = model.predict(final_features)

    output = round(prediction[0], 9)

    return render_template('index.html', prediction_text='PM2.5 value should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)