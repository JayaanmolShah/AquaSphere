import uuid
from django.db import models

class irrig_sched(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  
        editable=False
    )
    crop_name = models.CharField(max_length=255)  
    date = models.DateField()  
    scheduled_irrigation_liters = models.FloatField()

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

    def __str__(self):
        return f"{self.crop_name} - {self.date}"