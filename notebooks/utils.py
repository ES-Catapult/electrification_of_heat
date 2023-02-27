import pandas as pd
import os


def load_data(path: str, file_format: str) -> pd.DataFrame:
    """Load data from file

    Args:
        path (str): Path to the file to load

    Returns:
        pd.DataFrame: The data
    """

    if file_format == "csv":
        data = pd.read_csv(f"{path}.csv")
        data = data.rename({"Timestamp": "timestamp"}, axis=1).set_index("timestamp")
        data["sensor_type"] = data["sensor_type"].str.lower()
    elif file_format == "parquet":
        data = pd.read_parquet(path)

    # Rename heat pump energy consumed to whole system energy consumed in case it is incorrectly labelled in the raw data
    data["sensor_type"] = data["sensor_type"].replace({"Heat_Pump_Energy_Consumed": "Whole_System_Energy_Consumed"})

    if "Timestamp" in data.columns:
        data = data.set_index("Timestamp")

    if "raw" in path:
        # Raw data needs to be adjusted a bit

        # Data has changed the capitalisation of the sensor types so we change it back to match the code
        sensor_list = [
            "Brine_Return_Temperature",
            "Heat_Pump_Return_Temperature",
            "Heat_Pump_Heating_Flow_Temperature",
            "Internal_Air_Temperature",
            "Boiler_Energy_Output",
            "Whole_System_Energy_Consumed",
            "Immersion_Heater_Energy_Consumed",
            "Back-up_Heater_Energy_Consumed",
            "External_Air_Temperature",
            "Hot_Water_Flow_Temperature",
            "Circulation_Pump_Energy_Consumed",
            "Heat_Pump_Energy_Output",
            "Brine_Flow_Temperature",
        ]
        sensor_map = {sensor.lower(): sensor for sensor in sensor_list}
        # Convert lower-case sensor_type to expected upper-case first letters
        data["sensor_type"] = data["sensor_type"].map(lambda x: sensor_map[x])
        data.index.name = "Timestamp"

    if "raw" in path:
        # Data where temperatures haven't been set to correct units need to be adjusted

        # Avoid conflict from duplicate indexes by resetting index
        data = data.reset_index()

        data = data.set_index("Timestamp")

    data.index = pd.to_datetime(data.index)

    data = data.reset_index().sort_values(["sensor_type", "Timestamp"]).set_index("Timestamp")

    return data


def load_temperature_data(path: str, file_format: str) -> pd.DataFrame:
    """Load only the temperature data from file

    Args:
        path (str): The file path

    Returns:
        pd.DataFrame: The temperature data
    """

    data = load_data(path, file_format=file_format)

    temperature_data = filter_temperature_data(data)

    return temperature_data


def filter_temperature_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter the data to just temperature data sensors

    Args:
        data (pd.DataFrame): The data to filter

    Returns:
        pd.DataFrame: The temperature data
    """

    temperature_vars = [sensor for sensor in data["sensor_type"].unique() if sensor.endswith("Temperature")]
    sensor_mask = data["sensor_type"].isin(temperature_vars)
    temperature_data = data.loc[sensor_mask].copy()

    return temperature_data


def resample_data(data: pd.DataFrame, freq: str = "2T") -> pd.DataFrame:
    """Resample the data to specified frequency
    Cumulative and non-cumulative sensors will be aggregated using max and mean functions respectively

    Args:
        data (pd.DataFrame): The data to be resampled

    Returns:
        pd.DataFrame: The resampled data
    """

    cumulative_sensors = [
        "Boiler_Energy_Output",
        "Whole_System_Energy_Consumed",
        "Immersion_Heater_Energy_Consumed",
        "Back-up_Heater_Energy_Consumed",
        "Circulation_Pump_Energy_Consumed",
        "Heat_Pump_Energy_Output",
    ]

    non_cumulative_sensors = [
        "Brine_Return_Temperature",
        "Heat_Pump_Return_Temperature",
        "Heat_Pump_Heating_Flow_Temperature",
        "Internal_Air_Temperature",
        "External_Air_Temperature",
        "Hot_Water_Flow_Temperature",
        "Brine_Flow_Temperature",
    ]

    cumulative_data = data.loc[data["sensor_type"].isin(cumulative_sensors)]
    if len(cumulative_data) > 0:
        cumulative_data = cumulative_data.groupby("sensor_type").resample(freq).max()
        if "sensor_type" in cumulative_data.columns:
            cumulative_data = cumulative_data.droplevel(0)
        else:
            cumulative_data = cumulative_data.reset_index(level=0)

    non_cumulative_data = data.loc[data["sensor_type"].isin(non_cumulative_sensors)]

    if len(non_cumulative_data) > 0:
        non_cumulative_data = non_cumulative_data.groupby("sensor_type").resample(freq).mean()
        if "sensor_type" in non_cumulative_data.columns:
            non_cumulative_data = non_cumulative_data.droplevel(0)
        else:
            non_cumulative_data = non_cumulative_data.reset_index(level=0)

    data = pd.concat([cumulative_data, non_cumulative_data])

    return data

def create_folder_structure(eoh_folder):
    """Creates the folder structure required to run the code

    Args:
        eoh_folder (_type_): The top level directory

    Returns:
        _type_: Locations used with the code
    """
    # Specify location of specific folders
    location = os.path.join(eoh_folder)
    location_in_raw = os.path.join(location, "raw")
    location_out = os.path.join(location, "processed")
    location_out_binned_temp = os.path.join(location_out, "binned_heating_temperature")
    location_out_binned_temp_plots = os.path.join(location_out_binned_temp, "plots")
    location_out_binned_temp_flow = os.path.join(location_out_binned_temp, "all_flow_temperatures")
    location_out_binned_temp_plots_flow = os.path.join(location_out_binned_temp_plots, "all_flow_temperatures")
    location_out_cleaned = os.path.join(location_out, "cleaned")
    location_out_cleaned_data = os.path.join(location_out_cleaned, "cleaned")
    location_out_plots = os.path.join(location_out, "plots")
    location_out_temp_plots = os.path.join(location_out_plots, "temperature")
    location_out_cleaned_plots = os.path.join(location_out_cleaned, "plots")
    location_out_single_flagged_plots = os.path.join(location_out_cleaned_plots, "single_flagged")
    location_out_single_flagged_plots_corrected = os.path.join(location_out_single_flagged_plots, "corrected")
    location_out_single_flagged_plots_change_analysis = os.path.join(location_out_single_flagged_plots, "change_point_analysis")
    location_out_double_flagged_plots = os.path.join(location_out_cleaned_plots, "double_flagged")
    location_out_double_flagged_plots_corrected = os.path.join(location_out_double_flagged_plots, "corrected")
    location_out_full_plots = os.path.join(location_out_cleaned_plots, "full")
    location_out_window_plots = os.path.join(location_out_cleaned_plots, "window")
    location_out_cleaned_scored = os.path.join(location_out_cleaned, "cleaned_scored")

    # Make folders if they do not exist
    if not os.path.exists(location):
        os.mkdir(location)
    if not os.path.exists(location_in_raw):
        os.mkdir(location_in_raw)
    if not os.path.exists(location_out):
        os.mkdir(location_out)
    if not os.path.exists(location_out_cleaned):
        os.mkdir(location_out_cleaned)
    if not os.path.exists(location_out_cleaned_data):
        os.mkdir(location_out_cleaned_data)
    if not os.path.exists(location_out_plots):
        os.mkdir(location_out_plots)
    if not os.path.exists(location_out_temp_plots):
        os.mkdir(location_out_temp_plots)
    if not os.path.exists(location_out_cleaned_plots):
        os.mkdir(location_out_cleaned_plots)
    if not os.path.exists(location_out_single_flagged_plots):
        os.mkdir(location_out_single_flagged_plots)
    if not os.path.exists(location_out_single_flagged_plots):
        os.mkdir(location_out_single_flagged_plots)
    if not os.path.exists(location_out_single_flagged_plots_corrected):
        os.mkdir(location_out_single_flagged_plots_corrected)
    if not os.path.exists(location_out_single_flagged_plots_change_analysis):
        os.mkdir(location_out_single_flagged_plots_change_analysis)
    if not os.path.exists(location_out_double_flagged_plots):
        os.mkdir(location_out_double_flagged_plots)
    if not os.path.exists(location_out_double_flagged_plots_corrected):
        os.mkdir(location_out_double_flagged_plots_corrected)
    if not os.path.exists(location_out_full_plots):
        os.mkdir(location_out_full_plots)
    if not os.path.exists(location_out_window_plots):
        os.mkdir(location_out_window_plots)
    if not os.path.exists(location_out_binned_temp):
        os.mkdir(location_out_binned_temp)
    if not os.path.exists(location_out_binned_temp_plots):
        os.mkdir(location_out_binned_temp_plots)
    if not os.path.exists(location_out_binned_temp_flow):
        os.mkdir(location_out_binned_temp_flow)
    if not os.path.exists(location_out_binned_temp_plots_flow):
        os.mkdir(location_out_binned_temp_plots_flow)
    if not os.path.exists(location_out_cleaned_scored):
        os.mkdir(location_out_cleaned_scored)
    
    return location, location_in_raw, location_out, location_out_cleaned