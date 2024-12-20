# Generated by Django 4.2.16 on 2024-11-30 18:09

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('home', '0004_remove_subscriber_campaign_delete_campaign_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='irrig_sched',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('crop_name', models.CharField(max_length=255)),
                ('date', models.DateField()),
                ('scheduled_irrigation_liters', models.FloatField()),
            ],
        ),
    ]
