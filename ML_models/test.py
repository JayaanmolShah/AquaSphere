# import pickle
# import pandas as pd
# from prophet import Prophet

# # Load the trained model
# with open('ML_models/prophet_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Sample data for prediction
# # Assuming the model expects a column 'ds' (date) and 'y' (target)
# future_data = pd.DataFrame({
#     'ds': pd.date_range(start='2024-01-01', periods=10, freq='D'),  # Example future dates
# })

# # Make predictions
# forecast = model.predict(future_data)

# # Print the forecast
# print(forecast[['ds', 'yhat']])  # yhat is the predicted value





# test 2
# import pickle
# import numpy as np
# import pandas as pd

# # Load the saved models
# print("Loading models...")
# with open("ML_models/water_level_model.pkl", "rb") as rf_file:
#     random_forest_model = pickle.load(rf_file)



# # Generate dummy data
# print("Generating dummy data...")
# dummy_data = pd.DataFrame({
#     'tavg': [20.5, 22.1, 18.7],  # Example features (make sure they match the training data)
#     'tmin': [15.2, 16.3, 14.5],
#     'tmax': [25.8, 28.1, 24.4],
#     'prcp': [0.0, 0.2, 0.1],
#     'downstream_water_level' :[700, 710, 720],
#     'inflow_rate': [500, 450, 600],
#     'outflow_rate': [400, 380, 420],
#     'Sum_Rainfall_Lag_3Days': [0.2, 0.3, 0.1],
#     'Sum_Rainfall_Lag_7Days': [1.0, 1.2, 0.8],
#     'Sum_Rainfall_Lag_14Days': [3.5, 3.8, 3.2],
#     'Sum_Rainfall_Lag_30Days': [5.0, 5.5, 4.9],
#     'Inflow_Lag_3Days': [1500, 1400, 1550],
#     'Inflow_Lag_7Days': [3000, 2800, 3100],
#     'Inflow_Lag_14Days': [7000, 6800, 7100],
#     'Inflow_Lag_30Days': [12000, 12500, 11900],
#     'Outflow_Lag_3Days': [1400, 1300, 1450],
#     'Outflow_Lag_7Days': [2800, 2700, 2900],
#     'Outflow_Lag_14Days': [6500, 6300, 6600],
#     'Outflow_Lag_30Days': [11500, 12000, 11300],
#     'evaporation_loss_mm': [2.0, 3.0, 3.5]
# })

# # Make predictions
# print("Making predictions...")
# rf_predictions = random_forest_model.predict(dummy_data)

# dummy_data['Random_Forest_Prediction'] = rf_predictions

# print("\nDummy Data Predictions:")
# print(dummy_data['Random_Forest_Prediction'].head())

# # Save predictions to a CSV file for review
# output_path = "dummy_predictions.csv"
# dummy_data.to_csv(output_path, index=False)
# print(f"Predictions saved to {output_path}.")



# test 3
import pickle
import numpy as np
import pandas as pd
import xgboost
# Load the saved models
print("Loading models...")
with open("ML_models/water_level_model.pkl", "rb") as rf_file:
    random_forest_model = pickle.load(rf_file)

with open("ML_models/xgboost_model.pkl", "rb") as xgb_file:
    xgboost_model = pickle.load(xgb_file)

# Generate dummy data
print("Generating dummy data...")
dummy_data = pd.DataFrame({
    'tavg': [20.5, 22.1, 18.7],  # Example features (make sure they match the training data)
    'tmin': [15.2, 16.3, 14.5],
    'tmax': [25.8, 28.1, 24.4],
    'prcp': [0.0, 0.2, 0.1],
    'downstream_water_level' :[700, 710, 720],
    'inflow_rate': [500, 450, 600],
    'outflow_rate': [400, 380, 420],
    'Sum_Rainfall_Lag_3Days': [0.2, 0.3, 0.1],
    'Sum_Rainfall_Lag_7Days': [1.0, 1.2, 0.8],
    'Sum_Rainfall_Lag_14Days': [3.5, 3.8, 3.2],
    'Sum_Rainfall_Lag_30Days': [5.0, 5.5, 4.9],
    'Inflow_Lag_3Days': [1500, 1400, 1550],
    'Inflow_Lag_7Days': [3000, 2800, 3100],
    'Inflow_Lag_14Days': [7000, 6800, 7100],
    'Inflow_Lag_30Days': [12000, 12500, 11900],
    'Outflow_Lag_3Days': [1400, 1300, 1450],
    'Outflow_Lag_7Days': [2800, 2700, 2900],
    'Outflow_Lag_14Days': [6500, 6300, 6600],
    'Outflow_Lag_30Days': [11500, 12000, 11300],
    'evaporation_loss_mm': [2.0, 3.0, 3.5]
})

# Make predictions
print("Making predictions...")
rf_predictions = random_forest_model.predict(dummy_data)
xgb_predictions = xgboost_model.predict(dummy_data)

dummy_data['Random_Forest_Prediction'] = rf_predictions
dummy_data['XGBoost_Prediction'] = xgb_predictions

print("\nDummy Data Predictions:")
print(dummy_data[['Random_Forest_Prediction', 'XGBoost_Prediction']].head())

# Save predictions to a CSV file for review
output_path = "data/dummy_predictions.csv"
dummy_data.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}.")