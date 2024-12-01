from django.contrib import admin
from .models import irrig_sched,water_requi_dummy,water_requi
# Register your models here.

admin.site.register(irrig_sched)
admin.site.register(water_requi_dummy)
admin.site.register(water_requi)
