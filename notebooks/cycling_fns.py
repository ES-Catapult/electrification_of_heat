import pandas as pd
import numpy as np


def prep_readings(readings, winter_only=True):
    """
    Prepares and processes heat pump consumption readings for analysis.

    This function performs several preprocessing steps on the input data:
    - Converts the 'Timestamp' column to a datetime format and sets it as the index.
    - Reindexes the data to ensure a complete time series with 2-minute intervals, filling gaps with NaN.
    - Computes power values in watts by differencing cumulative energy consumption.
    - Optionally filters data to include only winter months (December, January, and February).
    - Identifies when power consumption exceeds a given threshold (power_floor) and smooths transient switching states.
    - Groups consecutive periods of similar states for further analysis.

    Args:
        readings (pd.DataFrame): A DataFrame containing energy readings with at least the following columns:
            - 'Timestamp': The timestamp of each reading.
            - 'Whole_System_Energy_Consumed': Cumulative energy consumption in kWh.
            - 'Heat_Pump_Energy_Output': Cumulative energy output in kWh.
        winter_only (bool, optional): Whether to filter data to only winter months (December, January, February). Defaults to True.

    Returns:
        pd.DataFrame: Processed DataFrame with additional computed columns:
            - 'Power_W': Instantaneous power in watts, derived from cumulative consumption.
            - 'Heat_Pump_Power_Output': Instantaneous heat pump power output in watts.
    """
    readings = readings.copy()
    # Ensure Timestamp is in datetime format
    readings["Timestamp"] = pd.to_datetime(readings["Timestamp"], utc=True)

    # Set Timestamp as index
    readings.set_index("Timestamp", inplace=True)

    # Create a full range of timestamps every 2 minutes
    full_range = pd.date_range(
        start=readings.index.min(), end=readings.index.max(), freq="2T"
    )

    # Reindex to fill missing timestamps with NaN
    readings = readings.reindex(full_range)

    # Do not interpolate missing energy values, keep NaN

    # Convert cumulative kWh to non-cumulative kWh (differencing)
    readings["Power_W"] = readings["Whole_System_Energy_Consumed"].diff() * 30 * 1000
    readings["Heat_Pump_Power_Output"] = (
        readings["Heat_Pump_Energy_Output"].diff() * 30 * 1000
    )

    # Filter to Winter data only
    if winter_only:
        readings = readings[readings.index.month.isin([1, 2, 12])]

    return readings


def flag_cycle_groups(readings, power_floor=200):
    """
    Flags consecutive on/off periods that might consitute a cycle 
    - Identifies when power consumption exceeds a given threshold (power_floor) and smooths transient switching states.
    - Also treats drops of >5C in flow temperature as turning off
    - Groups consecutive periods of similar states for further analysis.

    Args:
        readings (pd.DataFrame): A DataFrame containing energy readings with at least the following columns:
            - 'Timestamp': The timestamp of each reading.
            - 'Power_W': Instantaneous power in watts, derived from cumulative consumption.
   
    Returns:
        pd.DataFrame: Processed DataFrame with additional computed columns:
            - 'on': Binary indicator for whether power consumption exceeds `power_floor`.
            - 'group': Identifier for consecutive periods of the same state.
            - 'position': Position within each identified state group.
    """

    # Create the "on" column
    readings["on"] = (readings["Power_W"] >= power_floor).astype(int)

    # If the flow temperature has dropped more than 5C relative to the previous two periods then treat it as an 'off'
    readings.loc[
        (readings["on"] == 1)
        & (
            (
                readings["Heat_Pump_Heating_Flow_Temperature"]
                - (
                    readings["Heat_Pump_Heating_Flow_Temperature"].shift(1)
                    + readings["Heat_Pump_Heating_Flow_Temperature"].shift(2)
                )
                / 2
            )
            <= -5
        ),
        "on",
    ] = 0

    # If the flow minus return temperature is less than 1C, treat it as an 'off'
    readings.loc[
        (readings["on"] == 1)
        & (
            readings["Heat_Pump_Heating_Flow_Temperature"]
            - readings["Heat_Pump_Return_Temperature"]
            <= 0.2
        ),
        "on",
    ] = 0

    # Smooth transient states where a single 'on' period is surrounded by 'off'
    # Single 'off' states are more likely to be real, so leave those alone
    readings.loc[
        (readings["on"] == 1)
        & (readings["on"].shift(1) == 0)
        & (readings["on"].shift(-1) == 0),
        "on",
    ] = 0

    # Identify groups of consecutive states
    readings["group"] = (
        readings["on"].fillna(-1) != readings["on"].fillna(-1).shift()
    ).cumsum()
    readings["position"] = readings.groupby("group").cumcount() + 1

    return readings


def calc_max_power(readings, percentile=95, power_floor=200, power_ceiling=5000):
    """
    Calculates the maximum power for a heat pump based on a given percentile of valid power readings.

    Args:
        readings (pd.DataFrame): A DataFrame containing a 'Power_W' column with power readings in watts.
        percentile (float, optional): The percentile value to compute for the filtered power readings.
                                      Defaults to 95 (i.e., the 95th percentile).
        power_floor (int, optional): The minimum valid power reading (in watts). Defaults to 100 W.
        power_ceiling (int, optional): The maximum valid power reading (in watts). Defaults to 5000 W.

    Returns:
        float: The computed power threshold at the specified percentile, rounded to the nearest 10 watts.
    """
    sensible_power = readings.loc[
        (readings["Power_W"] >= power_floor) & (readings["Power_W"] <= power_ceiling)
    ].copy()
    max_power = np.percentile(sensible_power["Power_W"], percentile).round(-1)
    return max_power


def identify_cycles(readings):
    """
    Identifies and summarizes operational cycles of an energy system based on power and temperature data.

    This function groups power readings into distinct operational cycles based on the 'group' identifier,
    which represents consecutive periods of similar system states. It then calculates various statistical
    metrics for each cycle, including power consumption, heating temperatures, and hot water temperatures.

    Args:
        readings (pd.DataFrame): A DataFrame containing time-indexed power and temperature readings with 
            the group column to flag cycles.

    Returns:
        pd.DataFrame: A summary DataFrame where each row represents an identified cycle with the following columns:
            - 'state': The operational state of the system (on/off).
            - 'start_time': The start time of the cycle.
            - 'duration': The duration of the cycle in minutes.
            - 'max_power': The maximum power consumption during the cycle (W).
            - 'min_power': The minimum power consumption during the cycle (W).
            - 'mean_power': The average power consumption during the cycle (W).
            - 'std_power': The standard deviation of power consumption during the cycle.
            - 'max_heat_temp': The maximum heating flow temperature recorded during the cycle (°C).
            - 'min_heat_temp': The minimum heating flow temperature recorded during the cycle (°C).
            - 'median_heat_temp': The median heating flow temperature during the cycle (°C).
            - 'std_heat_temp': The standard deviation of heating flow temperature during the cycle.
            - 'max_hot_water_temp': The maximum hot water flow temperature recorded during the cycle (°C).
            - 'min_hot_water_temp': The minimum hot water flow temperature recorded during the cycle (°C).
            - 'median_hot_water_temp': The median hot water flow temperature during the cycle (°C).
            - 'std_hot_water_temp': The standard deviation of hot water flow temperature during the cycle.
            - 'hot_water': Boolean flag indicating whether hot water usage was detected in the cycle.
            - 'temp_change': The difference between maximum and minimum heating flow temperatures in the cycle.

    Notes:
        - Cycles where both 'max_power' and at least one of 'max_heat_temp' or 'max_hot_water_temp' are missing 
          are excluded from the final results.
        - The 'hot_water' column is set to True if any valid hot water temperature readings exist in a cycle.
        - The 'temp_change' column measures the temperature fluctuation during the heating cycle.
    """

    # Group by the identified state periods
    cycles = (
        readings.groupby("group")
        .agg(
            state=("on", "first"),
            start_time=("Power_W", lambda x: x.index.min()),
            duration=(
                "Power_W",
                lambda x: (x.index.max() - x.index.min()).total_seconds() / 60,
            ),
            max_power=("Power_W", "max"),
            min_power=("Power_W", "min"),
            mean_power=("Power_W", "mean"),
            std_power=("Power_W", "std"),
            max_heat_temp=("Heat_Pump_Heating_Flow_Temperature", "max"),
            min_heat_temp=("Heat_Pump_Heating_Flow_Temperature", "min"),
            median_heat_temp=("Heat_Pump_Heating_Flow_Temperature", "median"),
            mean_external_temp=("External_Air_Temperature", "mean"),
            mean_internal_temp=("Internal_Air_Temperature", "mean"),
            std_heat_temp=("Heat_Pump_Heating_Flow_Temperature", "std"),
            max_hot_water_temp=("Hot_Water_Flow_Temperature", "max"),
            min_hot_water_temp=("Hot_Water_Flow_Temperature", "min"),
            median_hot_water_temp=("Hot_Water_Flow_Temperature", "median"),
            std_hot_water_temp=("Hot_Water_Flow_Temperature", "std"),
            heat_temp_diff=(
                "Heat_Pump_Heating_Flow_Temperature",
                lambda x: x.iloc[-2] - x.iloc[1] if len(x) > 2 else None,
            ),
        )
        .reset_index(drop=True)
    )

    cycles["hot_water"] = cycles["max_hot_water_temp"].notna()

    cycles = cycles[
        cycles["max_power"].notna()
        & (cycles["max_heat_temp"].notna() | cycles["max_hot_water_temp"].notna())
    ]

    cycles["temp_change"] = cycles["max_heat_temp"] - cycles["min_heat_temp"]

    return cycles


def calc_cycling_features(cycles, max_power):
    """
    Computes key cycling metrics for heat pump based on identified operational cycles.

    This function extracts statistical features from the cycle data to quantify system behavior,
    including cycle durations, power modulation, and heating performance.

    Args:
        cycles (pd.DataFrame): A DataFrame summarizing system cycles
        max_power (float): The estimated maximum power output of the system.

    Returns:
        pd.Series: A Series containing key cycling metrics:
            - 'median_on_duration': Median duration of heating cycles (excluding hot water cycles) in minutes.
            - 'median_off_duration': Median duration of off cycles in minutes.
            - 'median_hw_duration': Median duration of hot water heating cycles in minutes.
            - 'median_flow_temp': Median heating flow temperature across heating cycles (excluding hot water).
            - 'median_temp_change': Median temperature change within heating cycles (excluding hot water).
            - 'mean_power': Mean power consumption during heating cycles (excluding hot water).
            - 'mean_modulation_pct': Mean power modulation as a percentage of `max_power`.
            - 'median_cycles_per_day': Median number of heating cycles per day.

    Notes:
        - Heating cycles are considered separately from hot water cycles.
    """

    median_on_duration = cycles.loc[
        (cycles["state"] == 1) & (~cycles["hot_water"]), "duration"
    ].median()
    median_off_duration = cycles.loc[cycles["state"] == 0, "duration"].median()
    median_hw_duration = cycles.loc[
        (cycles["state"] == 1) & (cycles["hot_water"]), "duration"
    ].median()
    median_flow_temp = cycles.loc[
        (cycles["state"] == 1) & (~cycles["hot_water"]), "median_heat_temp"
    ].median()
    median_temp_change = cycles.loc[
        (cycles["state"] == 1) & (~cycles["hot_water"]), "temp_change"
    ].median()
    median_heat_temp_diff = cycles.loc[
        (cycles["state"] == 1) & (~cycles["hot_water"]), "heat_temp_diff"
    ].median()
    mean_power = (
        cycles.loc[(cycles["state"] == 1) & (~cycles["hot_water"]), "mean_power"]
        .mean()
        .round(0)
    )
    mean_modulation_pct = mean_power / max_power * 100
    median_cycles_per_day = (
        cycles[cycles["state"] == 1]
        .groupby(cycles["start_time"].dt.date)
        .count()["state"]
        .median()
    )

    cycling_features = {
        "median_on_duration": median_on_duration,
        "median_off_duration": median_off_duration,
        "median_hw_duration": median_hw_duration,
        "median_flow_temp": median_flow_temp,
        "median_temp_change": median_temp_change,
        "median_heat_temp_diff": median_heat_temp_diff,
        "mean_power": mean_power,
        "mean_modulation_pct": mean_modulation_pct,
        "median_cycles_per_day": median_cycles_per_day,
    }
    cycling_features = pd.Series(cycling_features)
    return cycling_features


def get_all_features(readings, power_floor=200):
    readings = prep_readings(readings)
    readings = flag_cycle_groups(readings, power_floor=power_floor)
    max_power = calc_max_power(readings, power_floor=power_floor)
    cycles = identify_cycles(readings)
    cycling_features = calc_cycling_features(cycles, max_power)
    median_internal_temp_winter = readings["Internal_Air_Temperature"].median()
    tenth_pct_internal_temp_winter = np.percentile(
        readings["Internal_Air_Temperature"].dropna(), 10
    )
    percentage_time_on = (readings["Power_W"] > 200).mean()
    additional_features = pd.Series(
        {
            "median_internal_temp_winter": median_internal_temp_winter,
            "tenth_pct_internal_temp_winter": tenth_pct_internal_temp_winter,
            "percentage_time_on": percentage_time_on,
        }
    )
    cycling_features = pd.concat([cycling_features, additional_features])

    return cycling_features


def get_annual_features(readings, selected_window_start, selected_window_end):
    readings = prep_readings(readings, winter_only=False)
    readings = readings[selected_window_start:selected_window_end]
    readings["hot_water"] = readings["Hot_Water_Flow_Temperature"].notna()

    hot_water_usage = readings.loc[
        (readings["hot_water"]) & (readings["Heat_Pump_Power_Output"] > 0)
    ]
    heating_usage = readings.loc[
        ~(readings["hot_water"]) & (readings["Heat_Pump_Power_Output"] > 0)
    ]

    hot_water_energy_usage = hot_water_usage["Heat_Pump_Power_Output"].sum() / 30 / 1000
    heating_energy_usage = heating_usage["Heat_Pump_Power_Output"].sum() / 30 / 1000

    max_flow_temp = readings["Heat_Pump_Heating_Flow_Temperature"].max()
    flow_temp_99th_pct = np.percentile(
        readings["Heat_Pump_Heating_Flow_Temperature"].dropna(), 99
    )

    readings["combined_flow_temperature"] = readings[
        "Heat_Pump_Heating_Flow_Temperature"
    ].fillna(readings["Hot_Water_Flow_Temperature"])
    weighted_flow_temperature = (
        readings["combined_flow_temperature"] * readings["Heat_Pump_Power_Output"]
    ).mean() / readings["Heat_Pump_Power_Output"].mean()

    features = pd.Series(
        {
            "annual_hot_water_demand": hot_water_energy_usage,
            "annual_heating_demand": heating_energy_usage,
            "power_weighted_flow_temperature": weighted_flow_temperature,
            "max_flow_temp": max_flow_temp,
            "flow_temp_99th_pct": flow_temp_99th_pct,
        }
    )
    return features
