import pandas as pd
from prophet import Prophet
import pickle

def safe_convert(val):
    try:
        return pd.to_numeric(val)
    except (ValueError, TypeError):
        return val

def train_prophet_model(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    if df is None:
        raise ValueError(f"Failed to load data from {data_path}")
    print("Data loaded successfully:", df.head())  
    df = df.apply(safe_convert)
    df.infer_objects(copy=False)
    df.interpolate('linear', inplace=True)
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df.sort_values('measurement_date', inplace=True)

    
    # Prepare data for Prophet
    df_train = df.rename(columns={'measurement_date': 'ds', 'upstream_water_level': 'y'})
    df_train.drop(['downstream_water_level', 'inflow_rate', 'outflow_rate'], axis=1, inplace=True)

    # Train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_train)

    return model

def save_model(model, file_name='ML_models/prophet_model.pkl'):
    # Save the model to a file
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    model = train_prophet_model('D:/sih/backend/data/file1.csv')  # Replace 'file.csv' with your dataset
    save_model(model)
    print("Model trained and saved as prophet_model.pkl")
