from channels.generic.websocket import AsyncWebsocketConsumer
from .models import Alert
import json

class AlertConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def send_alerts(self, alerts):
        alerts = Alert.objects.filter(is_active=True).values()
        await self.send(text_data=json.dumps(list(alerts)))