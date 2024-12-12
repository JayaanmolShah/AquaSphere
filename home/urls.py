from django.urls import path
from home import views
from .views import ForecastAPIView,PredictWaterLevelAPIView, IrriSchedAPIView,temp_predictAPIView,PredictionAPIView,dashboardAPIView,Irrigation_schedulerAPIView,predictive_analysis_APIView,dam_dataAPIView
urlpatterns=[
    # path("",views.index,name='home'),
    path('forecast',ForecastAPIView.as_view(),name="forecast"),
    path('predict',PredictWaterLevelAPIView.as_view(),name="predict"),
    path('scheduler',IrriSchedAPIView.as_view(),name="scheduler"),
    path('temp_predict',temp_predictAPIView.as_view(),name="temp_predict"),
    path('all_predict',PredictionAPIView.as_view(),name="all_predict"),
    path('dashboard',dashboardAPIView.as_view(),name="dashboard"),
    path('irri_sch',Irrigation_schedulerAPIView.as_view(),name="irri_sch"),
    path('pred_analysis',predictive_analysis_APIView.as_view(),name="pred_analysis"),
    path('dam',dam_dataAPIView.as_view(),name="dam"),
]

