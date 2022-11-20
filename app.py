from flask import Flask,render_template,jsonify,request
import os
import joblib
import numpy as np

app=Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict' , methods=['POST','GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    x = np.array([[item_weight,item_fat_content,item_type,item_mrp,outlet_size,outlet_location_type,outlet_type]])
    scaler_path = r'E:\Sales-Forecasting\models\ss.sav'
    sc = joblib.load(scaler_path)
    x_std = sc.transform(x)

    model_path = r'E:\Sales-Forecasting\models\lr.sav'
    model = joblib.load(model_path)
    y_pred = model.predict(x_std)
    y_predf=int(y_pred)

    #final_result = jsonify({"Prediction":float(y_pred)})

    return render_template("predict.html" , data = y_predf)

    #return jsonify({"Prediction": float(y_pred)})

@app.route("/")
def redirect():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True,port=9456)

