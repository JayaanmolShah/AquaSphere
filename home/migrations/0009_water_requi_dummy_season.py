# Generated by Django 4.2.16 on 2024-12-01 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0008_water_requi_dummy'),
    ]

    operations = [
        migrations.AddField(
            model_name='water_requi_dummy',
            name='season',
            field=models.CharField(max_length=225, null=True),
        ),
    ]
