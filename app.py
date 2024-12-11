from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

#load model
model = joblib.load('ridge_model.pkl')

#define features
features = ['region', 'experience_level', 'job_category', 'company_size']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #get input
        input_data = {
            'region': request.form.get('region'),
            'experience_level': request.form.get('experience_level'),
            'job_category': request.form.get('job_category'),
            'company_size': request.form.get('company_size')
        }
        

        print("Received Input Data:", input_data)  #test print
        
        #somehow pd.get_dummies can't encode, do this manually
        categories = {
            'region': ['Americas', 'Asia', 'Australia and Oceania', 'Europe', 'Middle East'],
            'experience_level': ['EX', 'MI', 'SE'],
            'job_category': ['Data Scientist', 'Data Engineer', 'Machine Learning Specialist'],
            'company_size': ['M', 'S']
        }
        
        encoded_data = {}
        for feature, category_list in categories.items():
            for category in category_list:
                encoded_data[f"{feature}_{category}"] = int(input_data[feature] == category)

         
        input_encoded = pd.DataFrame([encoded_data])  #convert to data frame

        #allignment
        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[model.feature_names_in_]

        print("Aligned Input Encoded Data:", input_encoded)  #test print

        # model prediction
        temp = model.predict(input_encoded)
        prediction = np.expm1(temp[0])

        
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
