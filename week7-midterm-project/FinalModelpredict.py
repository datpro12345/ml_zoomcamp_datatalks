# Loading the Model - 
import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

# creating a variable with our model file:
input_file = 'model.bin'

# loads our model file: 
with open(input_file, 'rb') as f_in:    # file input; rb - used to read the file
    dv, model = pickle.load(f_in)     # load() function reads from the file

# creating a flask app -
app = Flask(__name__)

# adding a decorator to our function -
@app.route('/strokepredict/', methods=['POST'])

def strokepredict():
    # specifying request to be in JSON format converted to Python dictionary:
    sample_person = request.get_json()

    # transforming the sample person's feature details into a dictionary using DictVectorizer:
    X = dv.transform([sample_person])

    # converting the transformed feature matrix of sample person to DMatrix for use in XGBoost model:
    X_Dm = xgb.DMatrix(X, feature_names = dv.get_feature_names())

    # make prediction on sample person using our model: 
    y_pred = model.predict(X_Dm)

    # binary class value for stroke prediction, specifying our threshold as 0.55 for stroke decision:
    stroke = float(y_pred) >= 0.55

    # specify what response we want the web service to return to us:
    result = {'stroke_prediction': float(y_pred), 'stroke': bool(stroke)}
    
    return jsonify(result)

if __name__== "__main__":
    # running the commands in debug mode, specifying the host and port -
    app.run(debug=True)

