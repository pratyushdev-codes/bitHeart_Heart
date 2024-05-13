import pandas as pd
from flask import Flask,request, url_for, redirect, render_template
import joblib
import numpy as np

app = Flask(__name__)

model= joblib.load('model/model.pkl')
class_list = ["Normal", "Suspect", "Pathological"]
input_data = {
    "age":[0],
    "sex":[0],
    "cp":[0],
    "trestbps":[0], 
    "chol":[0], 
    "fbs":[0],
    "restecg":[0], 
    "thalach":[0],
    "exang":[0],
    "oldpeak":[0],
    "slope":[0], 
    "ca":[0],
    "thal":[0], 
    "target":[0], 
}


@app.route('/')
def home_page():
    return render_template("home.html")


@app.route('/predict', methods=['POST','GET'])
def predict():

    input_data = {
    "age":[0], 
    "sex":[0],
    "cp":[0],
    "trestbps":[0], 
    "chol":[0], 
    "fbs":[0],
    "restecg":[0], 
    "thalach":[0],
    "exang":[0],
    "oldpeak":[0],
    "slope":[0], 
    "ca":[0],
    "thal":[0], 
    "target":[0], 
   
    }

    prediction = ""
    
    if request.method == "POST":
        input_data["age"] = float(request.form['age'])
        input_data["sex"] = float(request.form['sex'])
        input_data["cp"] = float(request.form['cp'])
        input_data["trestbps"] = float(request.form['trestbps'])
        input_data["chol"] = float(request.form['chol'])
        input_data["fbs"] = float(request.form['fbs'])
        input_data["restecg"] = float(request.form['restecg'])
        input_data["thalach"] = float(request.form['thalach'])
        input_data["exang"] = float(request.form['exang'])
        input_data["oldpeak"] = float(request.form['oldpeak'])
        input_data["slope"] = float(request.form['slope'])
        input_data["ca"] = float(request.form['ca'])
        input_data["thal"] = float(request.form['thal'])
        input_data["target"] = float(request.form['target'])
       
        data = pd.DataFrame(input_data, index=[0])

        loaded_model = joblib.load('model/model.pkl')
        prediction = loaded_model.predict(data)

        return render_template('after.html', data=int(prediction))



if __name__ == '__main__':
    app.run(debug=True)