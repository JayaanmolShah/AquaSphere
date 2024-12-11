# Generated by Django 4.2.16 on 2024-12-11 13:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0018_alter_longtermoutput_date_alter_rainfalldata_date_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='dam_data',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(null=True)),
                ('lat', models.FloatField(null=True)),
                ('long', models.FloatField(null=True)),
            ],
        ),
        migrations.AddField(
            model_name='longtermoutput',
            name='silt',
            field=models.FloatField(null=True),
        ),
    ]