# Import necessary libraries
import pandas as pd
import joblib

# Now you can load the model whenever needed
loaded_model = joblib.load('crop_recommendation_model.joblib')

# Use the loaded model to make predictions
new_data = pd.DataFrame({'Type of Soil': [3], 'Season': [5], 'Rainfall in mm': [1000]})
prediction = loaded_model.predict(new_data)
print(f"Recommended Crop: {prediction[0]}")
