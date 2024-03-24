from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('z_house.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data['model']
le_location = data['le_location']
le_area_cat = data['le_area_cat']

@app.route('/')
def homepage():
    return render_template('zhouse.html')

#making predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    d1 = request.form['property_type']
    d2 = request.form['location']
    d3 = request.form['city']
    d4 = request.form['province_name']
    d5 = request.form['latitude']
    d6 = request.form['longitude']
    d7 = request.form['baths']
    d8 = request.form['purpose']
    d9 = request.form['bedrooms']
    d10 = request.form['Area Type']
    d11 = request.form['Area Size']
    d12 = request.form['Area Category']
    
    if d1 == 'House':
        d1=3
    if d1 == 'Flat':
        d1=2
    if d1 == 'Upper Portion':
        d1=1
    if d1 == 'Lower Portion':
        d1=0
        
    if d3 == 'Karachi':
        d3 = 4
    if d3 == 'Lahore':
        d3=3
    if d3 == 'Islamabad':
        d3=2
    if d3 == 'Rawalpindi':
        d3=1
    if d3 == 'Faisalabad':
        d3=0
        
    if d4 == 'Punjab':
        d4=2
    if d4 == 'Sindh':
        d4=1
    if d4 == 'Islamabad Capital':
        d4=0
        
    if d8 == 'For Sale':
        d8=1
    if d8 == 'For Rent':
        d8=0
        
    if d10 == 'Marla':
        d10 = 1
    if d10 == 'Kanal':
        d10 = 0
        
    user_input = np.array([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]])
    user_input[:,1] = le_location.transform(user_input[:,1])
    user_input[:,11] = le_area_cat.transform(user_input[:,11])
    user_input = user_input.astype(float)
    
    prediction = model.predict(user_input)
    prediction = np.round(prediction,2)
    
    text = f'The predicted house price is {prediction}'
    
    return render_template('zhouse.html',pred_text = text)

if __name__ == '__main__':
    app.run(debug=True)