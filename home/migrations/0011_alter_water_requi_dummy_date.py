# Generated by Django 4.2.16 on 2024-12-01 11:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0010_alter_water_requi_dummy_daily_wr_liters'),
    ]

    operations = [
        migrations.AlterField(
            model_name='water_requi_dummy',
            name='date',
            field=models.CharField(),
        ),
    ]
