# Generated by Django 4.2.16 on 2024-11-21 20:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_alter_campaign_slug'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='campaign',
            options={'ordering': ('-created_at',)},
        ),
        migrations.AlterModelOptions(
            name='subscriber',
            options={'ordering': ('-created_at',)},
        ),
    ]
