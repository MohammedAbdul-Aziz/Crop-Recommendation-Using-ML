from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scalers
model = pickle.load(open("model.pkl", "rb"))
mx = pickle.load(open("minmaxscaler.pkl", "rb"))
sc = pickle.load(open("standscaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get data from form and convert to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply preprocessing - IMPORTANT: Use transform, NOT fit_transform
        mx_features = mx.transform(single_pred)
        sc_features = sc.transform(mx_features)

        # Predict
        prediction = model.predict(sc_features)
        
        # Map prediction to crop name - ensure prediction is handled as an integer
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        
        # Convert prediction to integer and get crop
        pred_class = int(prediction[0])
        crop = crop_dict.get(pred_class)

        if crop:
            result = f"{crop} is the best crop to be cultivated right there"
        else:
            result = f"Sorry, we could not determine the best crop for class {pred_class}."
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print full error details to console for debugging
        result = f"Error during prediction: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)