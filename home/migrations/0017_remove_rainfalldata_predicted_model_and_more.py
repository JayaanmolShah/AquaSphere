# Generated by Django 4.2.16 on 2024-12-09 15:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0016_alter_predmodeloutputs_evapotranspiration_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='rainfalldata',
            name='predicted_model',
        ),
        migrations.RemoveField(
            model_name='temperaturedata',
            name='predicted_model',
        ),
    ]
