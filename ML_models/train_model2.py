import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error,r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def calculate_evaporation(tavg, tmax, tmin, inflow_rate):
    G = 0
    gamma = 0.066
    delta = (4098 * (0.6108 * np.exp((17.27 * tavg) / (tavg + 237.3)))) / ((tavg + 237.3) ** 2)

    Rn = inflow_rate * 0.0864
    es_tmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    vpd = es * 0.3
    evaporation = (0.408 * delta * (Rn - G) + gamma * (900 / (tavg + 273)) * 2 * vpd) / (delta + gamma)
    return max(evaporation, 0)

# Load your datasets
df1 = pd.read_csv("D:/sih/backend/data/weather.csv")
df2 = pd.read_csv("D:/sih/backend/data/level.csv")

# Data preprocessing
df1.interpolate('linear', inplace=True)
df2.interpolate('linear', inplace=True)

df2.rename(columns={"measurement_date": "date"}, inplace=True)
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])

data = pd.merge(df1, df2, on='date', how='inner')
data.drop(["snow", "wdir", "wspd", "wpgt", "pres", "tsun"], axis=1, inplace=True)

data['Sum_Rainfall_Lag_3Days'] = data['prcp'].rolling(window=3).sum()
data['Sum_Rainfall_Lag_7Days'] = data['prcp'].rolling(window=7).sum()
data['Sum_Rainfall_Lag_14Days'] = data['prcp'].rolling(window=14).sum()
data['Sum_Rainfall_Lag_30Days'] = data['prcp'].rolling(window=30).sum()
data['Inflow_Lag_3Days'] = data['inflow_rate'].rolling(window=3).sum()
data['Inflow_Lag_7Days'] =data['inflow_rate'].rolling(window=7).sum()
data['Inflow_Lag_14Days'] =data['inflow_rate'].rolling(window=14).sum()
data['Inflow_Lag_30Days'] =data['inflow_rate'].rolling(window=30).sum()
data['Outflow_Lag_3Days'] = data['outflow_rate'].rolling(window=3).sum()
data['Outflow_Lag_7Days'] =data['outflow_rate'].rolling(window=7).sum()
data['Outflow_Lag_14Days'] =data['outflow_rate'].rolling(window=14).sum()
data['Outflow_Lag_30Days'] =data['outflow_rate'].rolling(window=30).sum()
data.dropna(inplace=True)

surface_area_km2 = 1084  # Surface area of the reservoir in km² (Three Gorges Dam)
surface_area_m2 = surface_area_km2 * 1e6
data['evaporation_loss_mm'] = data.apply(
    lambda row: calculate_evaporation(row['tavg'], row['tmax'], row['tmin'], row['inflow_rate']), axis=1
)/1000

# Define features and target
X = data.drop(['upstream_water_level'], axis=1)
y = data['upstream_water_level']

# Train/test split
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(X_train.columns)  # Print all column names in X_train

X_train.drop('date', axis = 1, inplace = True)
X_test.drop('date', axis = 1, inplace = True)
# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=13)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"R^2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'max_features': ['sqrt', 'log2',None],  # Number of features to consider at each split
}

rf = RandomForestRegressor()
randomized_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
randomized_search.fit(X_train, y_train)

best = randomized_search.best_estimator_
res = best.predict(X_test)
print(f"best parameters: {randomized_search.best_params_}")
print(f"best score: {randomized_search.best_score_}")
print(f"best rmse {root_mean_squared_error(y_test,res)}")
print(f"mae {mean_absolute_error(y_test,res)}")
print(r2_score(y_test,res))

print(f"best parameters: {randomized_search.best_params_}")
print(f"best score: {randomized_search.best_score_}")
print(f"best rmse {root_mean_squared_error(y_test,res)}")
print(f"mae {mean_absolute_error(y_test,res)}")

# Save the model to a pickle file
with open("ML_models/water_level_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'water_level_model.pkl'")
