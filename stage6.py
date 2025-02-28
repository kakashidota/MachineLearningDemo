import pandas as pd
import joblib

model = joblib.load('music-recomender.joblib')

input_data = pd.DataFrame([[22, 0]], columns=['age', 'gender'])

prediction = model.predict(input_data)

print(f"Predicted music genre: {prediction[0]}")