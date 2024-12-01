from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.serializers import serialize
from .serializers import PredictionInputSerializer,WaterRequirementsSerializer,IrrigationSchedulesSerializer
from .models import irrig_sched,water_requi,water_requi_dummy
from .utils import generate_irrigation_schedule
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


# def index(request):
#     places = Place.objects.all()
#     places_geojson = serialize('geojson', places, geometry_field='location')
#     return JsonResponse(places_geojson, safe=False)

# Define a class-based view for predictions
class PredictWaterLevelAPIView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            with open("ML_models/water_level_model.pkl", "rb") as rf_file:
                random_forest_model = pickle.load(rf_file)

            with open("ML_models/xgboost_model.pkl", "rb") as xgb_file:
                xgboost_model = pickle.load(xgb_file)

            serializer = PredictionInputSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                # Convert input data to numpy array for prediction
            input_data = np.array([[
                serializer.validated_data['tavg'],
                serializer.validated_data['tmin'],
                serializer.validated_data['tmax'],
                serializer.validated_data['prcp'],
                serializer.validated_data['downstream_water_level'],
                serializer.validated_data['inflow_rate'],
                serializer.validated_data['outflow_rate'],
                serializer.validated_data['Sum_Rainfall_Lag_3Days'],
                serializer.validated_data['Sum_Rainfall_Lag_7Days'],
                serializer.validated_data['Sum_Rainfall_Lag_14Days'],
                serializer.validated_data['Sum_Rainfall_Lag_30Days'],
                serializer.validated_data['Inflow_Lag_3Days'],
                serializer.validated_data['Inflow_Lag_7Days'],
                serializer.validated_data['Inflow_Lag_14Days'],
                serializer.validated_data['Inflow_Lag_30Days'],
                serializer.validated_data['Outflow_Lag_3Days'],
                serializer.validated_data['Outflow_Lag_7Days'],
                serializer.validated_data['Outflow_Lag_14Days'],
                serializer.validated_data['Outflow_Lag_30Days'],
                serializer.validated_data['evaporation_loss_mm']
            ]])

                # Make predictions using both models
            rf_prediction = random_forest_model.predict(input_data)
            xgb_prediction = xgboost_model.predict(input_data)

            # Return predictions as a JSON response
            response_data = {
                'Random_Forest_Prediction': rf_prediction[0],
                'XGBoost_Prediction': xgb_prediction[0]
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ForecastAPIView(APIView):
    def post(self, request):
        try:
            # Load the trained model
            with open('ML_models/prophet_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Extract periods for future prediction from request data
            periods = int(request.data.get('periods', 10))  # Default to 10 if not provided

            # Prepare the future data for prediction
            future_data = pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=periods, freq='D')  # Modify this as per your requirement
            })

            # Make predictions
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