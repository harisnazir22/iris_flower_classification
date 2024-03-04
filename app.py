from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    DT = pickle.load(f)

# Define the category labels
Catagory = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    features = np.array([[float(x) for x in request.form.values()]])

    # Make prediction using the loaded model
    prediction = DT.predict(features)
    predicted_class = Catagory[int(prediction[0])]  # Convert predicted class index to label

    # Render the result template with the predicted class
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
