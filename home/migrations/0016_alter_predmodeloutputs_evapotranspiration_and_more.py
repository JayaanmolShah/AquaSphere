# Generated by Django 4.2.16 on 2024-12-09 07:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0015_predmodeloutputs_season_temperaturedata_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predmodeloutputs',
            name='evapotranspiration',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='predmodeloutputs',
            name='predicted_rain',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='predmodeloutputs',
            name='water_level_prophet',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='predmodeloutputs',
            name='water_level_xgb',
            field=models.FloatField(null=True),
        ),
        migrations.CreateModel(
            name='water_level_prophetreg',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('water_level_prophetreg', models.FloatField()),
                ('date', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='water_level_prophetreg', to='home.predmodeloutputs', to_field='date')),
            ],
        ),
    ]