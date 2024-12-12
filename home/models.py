import uuid
from django.db import models

class irrig_sched(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  
        editable=False
    )
    date = models.DateField()  
    total_water_requi_day = models.FloatField(null=True)
    avai_water=models.FloatField(null=True)
    diff= models.FloatField(null=True)

    def __str__(self):
        return f"{self.crop_name} - {self.date}"
    
class water_requi_dummy(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  
        editable=False
    )
    crop_name = models.CharField(max_length=255) 
    date = models.CharField() 
    season=models.CharField(max_length=225,null=True)
    daily_wr_mm = models.FloatField() 
    daily_wr_liters = models.FloatField(null=True)  
    rainfall_mm = models.FloatField()  
    area_ha = models.FloatField(null=True)

    def __str__(self):
        return f"{self.crop_name} - {self.date}"
    
class water_requi(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  
        editable=False
    )
    crop_name = models.CharField(max_length=255) 
    date = models.DateField()  
    daily_wr_mm = models.FloatField() 
    daily_wr_liters = models.FloatField() 
    rainfall_mm = models.FloatField() 
    area_ha = models.FloatField(null=True) 
    daily_idustrial_requi=models.FloatField(null=True)
    daily_domestic_requi=models.FloatField(null=True) 
 


    def __str__(self):
        return f"{self.crop_name} - {self.date}"
    
class Alert(models.Model):
    type = models.CharField(max_length=50)  # 'flood' or 'drought'
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

class sensor_data(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  
        editable=False
    )
    timestamp=models.DateTimeField(auto_now_add=True)
    soil_moisture = models.FloatField()  # Float field for soil moisture
    temperature = models.FloatField()  # Float field for temperature
    npk_values = models.TextField()  # Text field for NPK sensor data

    def __str__(self):
        return f"SensorData(id={self.id}, timestamp={self.timestamp})"
    

class Season(models.Model):
    id = models.BigAutoField(primary_key=True)
    season_name = models.TextField(null=False)
    start_date = models.DateField(null=False)
    end_date = models.DateField(null=False)

    def __str__(self):
        return self.season_name


class Requirement(models.Model):
    id = models.BigAutoField(primary_key=True)
    average_req = models.FloatField(null=False)
    daily_req = models.FloatField(null=False)
    litre_req = models.FloatField(null=False)
    date = models.DateField(null=False)
    season = models.CharField(null=False)

    def __str__(self):
        return f"Requirement on {self.date}"


class PredModelOutputs(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=False, unique=True)
    predicted_rain = models.FloatField(null=True)
    evapotranspiration = models.FloatField(null=True)
    evaporation_loss = models.FloatField(null=True)
    water_level_xgb = models.FloatField(null=True)
    water_level_prophet = models.FloatField(null=True)
    t_avg = models.FloatField(null=False)
    t_min = models.FloatField(null=False)
    t_max = models.FloatField(null=False)


    def __str__(self):
        return f"Predicted Model Output for {self.date}"

class RainfallData(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=True)
    precipitation = models.FloatField(null=False)

    def __str__(self):
        return f"Rainfall Data on {self.date}"


class TemperatureData(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=False)
    t_avg = models.FloatField(null=False)
    t_min = models.FloatField(null=False)
    t_max = models.FloatField(null=False)

    def __str__(self):
        return f"Temperature Data on {self.date}"

class ShortTermOutput(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=False)
    water_level_avg = models.FloatField(null=False)
    water_in_litres = models.FloatField(null=False)
    water_difference = models.FloatField(null=False)

    def __str__(self):
        return f"Short Term Output for {self.date}"


class LongTermOutput(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=False)
    water_level_prophet = models.FloatField(null=False)
    water_in_litres = models.FloatField(null=False)
    req_litres = models.FloatField(null=True)
    req_mm = models.FloatField(null=True)
    actual_avia_water_mm=models.FloatField(null=True)
    water_difference = models.FloatField(null=True)
    silt=models.FloatField(null=True)
    def __str__(self):
        return f"Long Term Output for {self.date}"

class water_level_prophetreg(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(null=False)
    water_level_prophetreg=models.FloatField(null=False)

class dam_data(models.Model):
    id = models.BigAutoField(primary_key=True)
    name=models.CharField(null=True)
    lat=models.FloatField(null=True)
    long=models.FloatField(null=True)
    damn_area=models.FloatField(null=True)
# Adding foreign key relationships for RainfallData and TemperatureData

