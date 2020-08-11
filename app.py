from flask import Flask, render_template, request
from datetime import datetime

import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    #Fuel_Type_Diesel=0
    #Fuel_arr = [CNG, DIESEL, PETROL]
    #Seller_arr = [DEALER, INDIVIDUAL]
    #Transmission_arr = [AUTOMATIC, MANUAL]
    temp_arr = list()
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        
        
        Fuel_Type=request.form['Fuel_Type']
        if(Fuel_Type=='Petrol'):
            Fuel_arr = [0,0,1]
        elif(Fuel_Type=='Diesel'):
            Fuel_arr = [0,1,0]
        else:
            Fuel_arr = [1,0,0]
            
            
        a = datetime.today().year
        Year_old=a-Year
        
        
        Seller_Type=request.form['Seller_Type']
        if(Seller_Type=='Individual'):
            Seller_arr = [0,1]
        else:
            Seller_arr = [1,0]
            
            
        Transmission_Type=request.form['Transmission_Type']
        if(Transmission_Type=='Manual'):
            Transmission_arr = [0,1]
        else:
            Transmission_arr = [1,0]
            
        temp_arr = [Present_Price,Kms_Driven2,Owner,Year_old] + Fuel_arr + Seller_arr + Transmission_arr
        
        data = np.array([temp_arr])    
        prediction=model.predict(data)
        output=round(prediction[0],2)
        
        if output<0:
            return render_template('index.html',prediction_text="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You can sell the car at {} lacs".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)