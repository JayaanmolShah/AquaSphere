# Generated by Django 4.2.16 on 2024-12-12 10:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0024_dam_data_damn_area'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='longtermoutput',
            name='req_litres',
        ),
        migrations.RemoveField(
            model_name='longtermoutput',
            name='water_difference',
        ),
        migrations.RemoveField(
            model_name='predmodeloutputs',
            name='water_level_xgb',
        ),
    ]