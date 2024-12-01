# Generated by Django 4.2.16 on 2024-12-01 10:34

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0007_alter_water_requi_area_ha'),
    ]

    operations = [
        migrations.CreateModel(
            name='water_requi_dummy',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('crop_name', models.CharField(max_length=255)),
                ('date', models.DateField()),
                ('daily_wr_mm', models.FloatField()),
                ('daily_wr_liters', models.FloatField()),
                ('rainfall_mm', models.FloatField()),
                ('area_ha', models.FloatField(null=True)),
            ],
        ),
    ]
