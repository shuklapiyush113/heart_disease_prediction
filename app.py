import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle

app=Flask(__name__)
model =pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/prediction')
def output():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)
    print(output)
    if output ==1:
        val = 'The patient is likely to have heart disease'
    else:
        val ='The patient is not likely to have heart disease'

    return render_template('prediction.html',pre=val)

    

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)