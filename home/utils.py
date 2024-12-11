import datetime
import pandas as pd
from .models import irrig_sched, water_requi,PredModelOutputs
import numpy as np

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





# Helper function to determine the season
def get_season(date):
    if not isinstance(date, datetime.date):
        raise ValueError(f"Expected a datetime.date object, got {type(date)}")

    # Extract the month and day
    month_day = date.strftime("%m-%d")
    print(f"Extracted Month-Day: {month_day}")  # Debugging

    # Check the season
    if "06-01" <= month_day <= "10-31":
        return "kharif"
    elif "11-01" <= month_day or month_day <= "03-31":  # Rabi spans year-end
        return "rabi"
    elif "04-01" <= month_day <= "05-31":
        return "summer"
    else:
        raise ValueError(f"Invalid date for season determination: {date}.")

# Function to calculate evaporation
def calculate_evaporation(tmin, tmax, tavg, season):
    G = 0  # Ground heat flux, negligible for reservoirs
    gamma = 0.066  # Psychrometric constant (kPa/°C)
    delta = (4098 * (0.6108 * np.exp((17.27 * tavg) / (tavg + 237.3)))) / ((tavg + 237.3) ** 2)

    es_tmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2  # Saturation vapor pressure (kPa)
    vpd = es * 0.3  # Vapor pressure deficit (kPa)

    # Assign approximate net radiation based on agricultural season
    if season == "kharif":
        Rn = 12  # MJ/m²/day (monsoon period)
    elif season == "rabi":
        Rn = 14  # MJ/m²/day (winter period)
    elif season == "summer":
        Rn = 18  # MJ/m²/day (pre-monsoon period)
    else:
        raise ValueError("Invalid season. Choose 'kharif', 'rabi', or 'summer'.")

    evaporation_loss = (0.408 * delta * (Rn - G) + gamma * (900 / (tavg + 273)) * 2 * vpd) / (delta + gamma)
    return max(evaporation_loss, 0)

# Main function to update the evaporation column
def update_evaporation():
    records = PredModelOutputs.objects.filter(evaporation_loss__isnull=True)
    for record in records:
        try:
            if not isinstance(record.date, datetime.date):
                raise TypeError(f"Expected datetime.date, got {type(record.date)} for record ID {record.id}")
            print(f"Record ID: {record.id}, Date: {record.date}, Extracted Month-Day: {record.date.strftime('%m-%d')}")
            # Retrieve required fields
            tmin = record.t_min  # Replace with actual field name for tmin
            tmax = record.t_max  # Replace with actual field name for tmax
            tavg = record.t_max   # Replace with the actual calculation if different
            season = get_season(record.date)
            # Calculate evaporation
            evaporation_loss = calculate_evaporation(tmin, tmax, tavg, season)
            
            # Update record
            record.evaporation_loss = evaporation_loss
            record.save()
        except Exception as e:
            print(f"Error processing record {record.id}: {e}")

# Call the function
