import pickle
from flask import Flask, request, jsonify


client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

with open('dv.bin', 'rb') as file:
    dv = pickle.load(file)

with open('model1.bin', 'rb') as file:
    model = pickle.load(file)

app = Flask('credit_card')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    y_pred = model.predict_proba(dv.transform(client))[0, 1]
    result = {
        'get_cc_probability': float(y_pred),
        'gets_cc': bool(y_pred >= 0.5),
    }
    return(jsonify(result))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)