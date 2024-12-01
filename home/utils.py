from datetime import datetime
import pandas as pd
from .models import irrig_sched, water_requi

def calculate_water_requirement_with_rainfall(crop_data, plantation_dates, rainfall_data,year):
    """
    Calculate water requirements adjusted for rainfall.
    """
    results = []
    year = int(year)

    for _, row in crop_data.iterrows():
        crop_name = row["crop_name"]
        crop_area = row["area_ha"]  # in hectares
        wr_mm_per_day = row["daily_wr_mm"]  # Water requirement in mm
        season = row["season"]

        # Get plantation period

        # Get plantation period using the provided year
        start_date = plantation_dates[season]["start"]
        end_date = plantation_dates[season]["end"]

        start_date = start_date.replace(year=year)
        end_date = end_date.replace(year=year)

        # Filter rainfall data for the crop's growing period
        crop_rainfall_data = rainfall_data[
            (rainfall_data["Date"] >= start_date) & (rainfall_data["Date"] <= end_date)
        ]
        

        for _, rain_row in crop_rainfall_data.iterrows():
            date = rain_row["Date"]
            rainfall_mm = rain_row["Rainfall_mm"]
            # Calculate daily water requirement
            daily_wr_mm = max(0, wr_mm_per_day - rainfall_mm)  # Ensure no negative water requirement
            daily_wr_liters = (daily_wr_mm * crop_area * 10)  # Convert mm to liters

            # Save to results
            results.append(water_requi(
                crop_name=crop_name,
                date=date,
                daily_wr_mm=daily_wr_mm,
                daily_wr_liters=daily_wr_liters,
                rainfall_mm=rainfall_mm
            ))

    # Bulk insert into the database
    water_requi.objects.bulk_create(results)

    return results

def generate_irrigation_schedule(crop_data, plantation_dates, rainfall_data,year):
    """
    Generate irrigation schedules adjusted for daily rainfall.
    """
    water_requirements = calculate_water_requirement_with_rainfall(crop_data, plantation_dates, rainfall_data,year)

    # Create irrigation schedule for days with unmet water needs
    irrigation_schedule = [
        irrig_sched(
            crop_name=req.crop_name,
            date=req.date,
            scheduled_irrigation_liters=req.daily_wr_liters
        )
        for req in water_requirements if req.daily_wr_liters > 0
    ]

    # Bulk insert into the database
    irrig_sched.objects.bulk_create(irrigation_schedule)

    return irrigation_schedule