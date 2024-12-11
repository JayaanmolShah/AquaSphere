from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.serializers import serialize
from .serializers import PredictionInputSerializer,WaterRequirementsSerializer,IrrigationSchedulesSerializer
from .models import irrig_sched,water_requi,water_requi_dummy, Season,TemperatureData,PredModelOutputs
from .utils import generate_irrigation_schedule
import requests
import pickle
from joblib import load
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date

class PredictWaterLevelAPIView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            prophet_model= load('ML_models/water_level_prophet.pkl')
            xgboost_model= load('ML_models/water_level_xgbreg.pkl')

            serializer = PredictionInputSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                # Convert input data to numpy array for prediction
            input_data = np.array([[
                serializer.validated_data['tavg'],
                serializer.validated_data['tmin'],
                serializer.validated_data['tmax'],
                serializer.validated_data['prcp'],
                serializer.validated_data['inflow_rate'],
                serializer.validated_data['outflow_rate'],
                serializer.validated_data['Sum_Rainfall_Lag_3Days'],
                serializer.validated_data['Sum_Rainfall_Lag_7Days'],
                serializer.validated_data['Sum_Rainfall_Lag_14Days'],
                serializer.validated_data['Sum_Rainfall_Lag_30Days'],
                serializer.validated_data['evaporation_loss_mm']
            ]])
            current_date = date.today()
            date_input = pd.DataFrame({'ds': [current_date]})
            
            pro_prediction = prophet_model.predict(date_input)['yhat'].iloc[0]
            xgb_prediction = xgboost_model.predict(input_data) 

            
            response_data = {
                'pro_Prediction': float(pro_prediction),  # Ensure scalar is handled correctly
                'XGBoost_Prediction': float(xgb_prediction)
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {'error': str(e)}
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)

class ForecastAPIView(APIView):
    def post(self, request):
        try:
            with open('ML_models/prophet_model.pkl', 'rb') as f:
                model = pickle.load(f)

            periods = int(request.data.get('periods', 10))  # Default to 10 if not provided

            future_data = pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=periods, freq='D')  # Modify this as per your requirement
            })

            forecast = model.predict(future_data)

            # Return the prediction result
            result = forecast[['ds', 'yhat']].to_dict(orient='records')
            return Response(result, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class IrriSchedAPIView(APIView):
    def post(self, request):
        try:
            # Extract season name and year from the request
            season_name = request.data.get("season_name")
            date = request.data.get("date")

            if not season_name or not date:
                return Response({"error": "Both season_name and year are required."}, status=status.HTTP_400_BAD_REQUEST)

            # Validate year
            try:
                date = int(date)
            except ValueError:
                return Response({"error": "Invalid year format. Please provide a valid year."}, status=status.HTTP_400_BAD_REQUEST)

            # Plantation dates for each season
            plantation_dates = {
                "Kharif": {"start": "06-01", "end": "10-31"},
                "Rabi": {"start": "11-01", "end": "03-31"},
                "Summer": {"start": "04-01", "end": "05-31"}
            }

            # Validate season name
            if season_name not in plantation_dates:
                return Response({"error": f"Invalid season_name: {season_name}"}, status=status.HTTP_400_BAD_REQUEST)

            # Get plantation start and end dates for the specified year
            # Get plantation start and end dates for the specified year
            start_date_str = f"{date}-{plantation_dates[season_name]['start']}"
            end_date_str = f"{date}-{plantation_dates[season_name]['end']}"

            # Parse the date strings into datetime objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Fetch crop data from the database where season and year match the input
            crop_data_query = water_requi_dummy.objects.filter(season=season_name, date=date)
            if not crop_data_query.exists():
                return Response({"error": f"No crop data found for season: {season_name} and year: {date}"}, status=status.HTTP_404_NOT_FOUND)

            # Convert crop data to DataFrame
            crop_data = pd.DataFrame(list(crop_data_query.values("crop_name", "area_ha", "daily_wr_mm", "season")))

            # Example rainfall data (you can replace this with real data)
            rainfall_data = pd.DataFrame({
                "Date": pd.date_range(start=start_date, end=end_date),
                "Rainfall_mm": ([2, 0, 5, 3, 1] * 31)[:len(pd.date_range(start=start_date, end=end_date))]
            })
            year=date

            # Generate irrigation schedule
            generate_irrigation_schedule(crop_data, {season_name: {"start": start_date, "end": end_date}}, rainfall_data,year)

            # Query database for all records related to this season and year
            water_requirements = water_requi.objects.filter(date__range=[start_date, end_date])
            irrigation_schedule = irrig_sched.objects.filter(date__range=[start_date, end_date])

            # Serialize and return the calculated results
            water_serializer = WaterRequirementsSerializer(water_requirements, many=True)
            irrigation_serializer = IrrigationSchedulesSerializer(irrigation_schedule, many=True)

            return Response({
                "WaterRequirements": water_serializer.data,
                "IrrigationSchedules": irrigation_serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

class temp_predictAPIView(APIView):
    def post(self, request):
        try:
            start_date = request.data.get('start_date')
            end_date = request.data.get('end_date')
            if not start_date or not end_date:
                return Response({"error": "Both start_date and end_date are required."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return Response({"error": "Invalid date format. Use YYYY-MM-DD."}, status=status.HTTP_400_BAD_REQUEST)

            if start_date > end_date:
                return Response({"error": "start_date must be before or equal to end_date."}, status=status.HTTP_400_BAD_REQUEST)

            # Load the saved Prophet models
            try:
                model_tmin = load('ML_models/tmin_prophet.pkl')
                model_tmax = load('ML_models/tmax_prophet.pkl')
                model_tavg = load('ML_models/tavg_prophet.pkl')

            except FileNotFoundError:
                return Response({"error": "One or more model files are missing."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            
            # Generate future dates if required
            num_days = (end_date - start_date).days + 1
            future_dates = pd.date_range(start=start_date - timedelta(days=num_days), periods=num_days * 2, freq='D')

            # Create DataFrame for forecasting
            future_df = pd.DataFrame({'ds': future_dates})

            # Generate predictions
            forecast_tmin = model_tmin.predict(future_df)
            forecast_tmax = model_tmax.predict(future_df)
            forecast_tavg = model_tavg.predict(future_df)

            # Filter predictions for the specific date range
            predicted_temperatures = pd.DataFrame({
                'date': future_df['ds'],
                'tmin': forecast_tmin['yhat'],
                'tavg': forecast_tavg['yhat'],
                'tmax': forecast_tmax['yhat']
            })
            predicted_temperatures = predicted_temperatures[(predicted_temperatures['date'] >= start_date) & (predicted_temperatures['date'] <= end_date)]

            water_requi_instances = [TemperatureData( t_min=row['tmin'], t_avg=row['tavg'], t_max=row['tmax'],date=row['date'])for _, row in predicted_temperatures.iterrows()]
            TemperatureData.objects.bulk_create(water_requi_instances)
            
            return Response(predicted_temperatures.to_dict(orient='records'), status=status.HTTP_200_OK)        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class PredictionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        # Parse input dates
        global_start_date = request.data.get('start_date', "2020-01-01")
        global_end_date = request.data.get('end_date', "2025-12-31")
        try:
            start_date = datetime.strptime(global_start_date, '%Y-%m-%d')
            end_date = datetime.strptime(global_end_date, '%Y-%m-%d')
        except ValueError:
            return Response({"error": "Invalid date format. Use YYYY-MM-DD."}, status=status.HTTP_400_BAD_REQUEST)

        # Load Prophet models
        try:
            models = {
                "tmin": load('ML_models/tmin_prophet.pkl'),
                "tmax": load('ML_models/tmax_prophet.pkl'),
                "tavg": load('ML_models/tavg_prophet.pkl'),
                "water_level": load('ML_models/water_level_prophet.pkl'),
                "et": load('ML_models/et_prophet.pkl'),
                "rainfall": load('ML_models/rain_monthly_prophet.pkl'),
            }
        except Exception as e:
            return Response({"error": f"Error loading models: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Generate daily and monthly future dates
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')

        # Create DataFrames for prediction
        daily_df = pd.DataFrame({'ds': daily_dates})
        monthly_df = pd.DataFrame({'ds': monthly_dates})

        # Initialize predictions dictionary
        predictions = {
            "tmin": None,
            "tmax": None,
            "tavg": None,
            "water_level_prophet": None,
            "evapotranspiration": None,
            "predicted_rain": None,
        }

        # Generate predictions
        try:
            predictions["tmin"] = models["tmin"].predict(daily_df).set_index('ds')['yhat']
            predictions["tmax"] = models["tmax"].predict(daily_df).set_index('ds')['yhat']
            predictions["tavg"] = models["tavg"].predict(daily_df).set_index('ds')['yhat']
            predictions["water_level_prophet"] = models["water_level"].predict(daily_df).set_index('ds')['yhat']
            predictions["evapotranspiration"] = models["et"].predict(daily_df).set_index('ds')['yhat']
            predictions["predicted_rain"] = models["rainfall"].predict(monthly_df).set_index('ds')['yhat']
        except Exception as e:
            return Response({"error": f"Error generating predictions: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Merge daily predictions
        result_df = pd.DataFrame(index=daily_dates)
        result_df['tmin'] = predictions["tmin"]
        result_df['tmax'] = predictions["tmax"]
        result_df['tavg'] = predictions["tavg"]
        result_df['water_level_prophet'] = predictions["water_level_prophet"]
        result_df['evapotranspiration'] = predictions["evapotranspiration"]

        # Merge monthly rainfall into daily DataFrame
        monthly_rainfall = predictions["predicted_rain"].reindex(daily_dates, method='pad')
        result_df['predicted_rain'] = monthly_rainfall

        # Save predictions to Django model
        instances = []
        for date, row in result_df.iterrows():
            instance = PredModelOutputs(
                date=date,
                predicted_rain=row['predicted_rain'],
                evapotranspiration=row['evapotranspiration'],
                water_level_xgb=None,  # Placeholder for XGB predictions if added later
                water_level_prophet=row['water_level_prophet'],
                t_avg=row['tavg'],
                t_min=row['tmin'],
                t_max=row['tmax'],
            )
            instances.append(instance)

        # Bulk insert into database
        try:
            PredModelOutputs.objects.bulk_create(instances)
        except Exception as e:
            return Response({"error": f"Error saving predictions to database: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {
                "message": "Predictions generated and saved successfully.",
                "start_date": global_start_date,
                "end_date": global_end_date,
                "total_records": len(instances),
            },
            status=status.HTTP_201_CREATED,
        )

class dashboardAPIView(APIView):
    def post(self,request):
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            daily_dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
            monthly_dates = pd.date_range(start=start_date_str, end=end_date_str, freq='M') 
            try:
                models = {
                    "water_level": load('ML_models/water_level_prophet.pkl'),
                    "et": load('ML_models/et_prophet.pkl'),
                    "rainfall": load('ML_models/rain_monthly_prophet.pkl'),
                }
            except Exception as e:
                return Response({"error": f"Error loading models: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            daily_df = pd.DataFrame({'ds': daily_dates})
            monthly_df = pd.DataFrame({'ds': monthly_dates})
            predictions = {
                "water_level_prophet": None, 
                "evapotranspiration": None,
                "predicted_rain": None,
            }
            try:
                predictions["water_level_prophet"] = models["water_level"].predict(daily_df).set_index('ds')['yhat']
                predictions["evapotranspiration"] = models["et"].predict(daily_df).set_index('ds')['yhat']
                predictions["predicted_rain"] = models["rainfall"].predict(monthly_df).set_index('ds')['yhat']
            except Exception as e:
                return Response({"error": f"Error generating predictions: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Store predictions in DataFrame
            result_df = pd.DataFrame(index=daily_dates)
            result_df['water_level_prophet'] = predictions["water_level_prophet"]
            result_df['evapotranspiration'] = predictions["evapotranspiration"]

            monthly_rainfall = predictions["predicted_rain"].reindex(daily_dates, method='pad')
            result_df['predicted_rain'] = monthly_rainfall
            
            daily_predictions = {
                "date": [],
                "water_level_prophet": [],
                "evapotranspiration": [],
                "predicted_rain": []
            }
            for index, row in result_df.iterrows():
                daily_predictions["date"].append(index.strftime('%Y-%m-%d'))
                daily_predictions["water_level_prophet"].append(row['water_level_prophet'])
                daily_predictions["evapotranspiration"].append(row['evapotranspiration'])
                daily_predictions["predicted_rain"].append(row['predicted_rain'])
            # Calculate the total water level prediction for the current water level
            # (sum of past 30 days predictions + current month rainfall + evapotranspiration)
            current_water_level = result_df['water_level_prophet'].iloc[-1]  # Last predicted water level for today
            total_rainfall_past_30_days = result_df['predicted_rain'].sum()  # Total rainfall for the last 30 days
            evapotranspiration_sum = result_df['evapotranspiration'].sum()  # Sum of evapotranspiration for the last 30 days

            return Response(
                    {
                        "message": "Predictions generated and saved successfully.",
                        "current_water_level": current_water_level,
                        "total_rainfall_past_30_days": total_rainfall_past_30_days,
                        "evapotranspiration_sum": evapotranspiration_sum,
                        
                    },
                    status=status.HTTP_200_OK,
                )
        except Exception as e:
            error_message = {'error': str(e)}
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
        
    