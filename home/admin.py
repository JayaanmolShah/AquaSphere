from django.contrib import admin
from .models import irrig_sched,water_requi_dummy,water_requi,Alert,sensor_data,Season,Requirement,PredModelOutputs,RainfallData,TemperatureData,LongTermOutput,ShortTermOutput,water_level_prophetreg
# Register your models here.

admin.site.register(irrig_sched)
admin.site.register(water_requi_dummy)
admin.site.register(water_requi)
admin.site.register(Alert)
admin.site.register(sensor_data)
admin.site.register(Season)
admin.site.register(Requirement)
admin.site.register(PredModelOutputs)
admin.site.register(RainfallData)
admin.site.register(TemperatureData)
admin.site.register(ShortTermOutput)
admin.site.register(LongTermOutput)
admin.site.register(water_level_prophetreg)


