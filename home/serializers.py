from rest_framework import serializers
from .models import water_requi,irrig_sched,dam_data

class PredictionInputSerializer(serializers.Serializer):
    tavg = serializers.FloatField()
    tmin = serializers.FloatField()
    tmax = serializers.FloatField()
    prcp = serializers.FloatField()
    inflow_rate = serializers.FloatField()
    outflow_rate = serializers.FloatField()
    Sum_Rainfall_Lag_3Days = serializers.FloatField()
    Sum_Rainfall_Lag_7Days = serializers.FloatField()
    Sum_Rainfall_Lag_14Days = serializers.FloatField()
    Sum_Rainfall_Lag_30Days = serializers.FloatField()
    evaporation_loss_mm = serializers.FloatField()

class WaterRequirementsSerializer(serializers.ModelSerializer):
    class Meta:
        model = water_requi
        fields = '__all__'

class IrrigationSchedulesSerializer(serializers.ModelSerializer):
    class Meta:
        model = irrig_sched
        fields = '__all__'

class DamSerializer(serializers.ModelSerializer):
    class Meta:
        model = dam_data
        fields = ['id', 'name', 'lat', 'long', 'damn_area']
        