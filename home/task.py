from celery import shared_task
from .models import Alert

@shared_task
def check_weather_and_trigger_alert():
    data = fetch_weather_data('Rajasthan')
    features = extract_features(data)  # Extract necessary features for prediction
    risk = predict_risk(features)
    if risk in ['flood', 'drought']:
        Alert.objects.create(
            type=risk,
            message=f"Warning: {risk.capitalize()} risk detected.",
        )
