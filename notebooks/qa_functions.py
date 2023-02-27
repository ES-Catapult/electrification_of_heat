from turtle import home
import pandas as pd
import numpy as np
import datetime
import requests
import json
import os
import matplotlib.pyplot as plt
import math
import data_scoring_fns as ds
from typing import Tuple

# from dateutil.relativedelta import relativedelta
# from sympy import denom

# Define the columns required for each metric
spfh_required_cols = {
    "spfh2": [
        "Heat_Pump_Energy_Output",
        "Whole_System_Energy_Consumed",
        "Back-up_Heater_Energy_Consumed",
        "Immersion_Heater_Energy_Consumed",
        "Circulation_Pump_Energy_Consumed",
        "Boiler_Energy_Output",
    ],
    "spfh3": [
        "Heat_Pump_Energy_Output",
        "Whole_System_Energy_Consumed",
        "Back-up_Heater_Energy_Consumed",
        "Immersion_Heater_Energy_Consumed",
        "Circulation_Pump_Energy_Consumed",
        "Boiler_Energy_Output",
    ],
    "spfh4": [
        "Heat_Pump_Energy_Output",
        "Whole_System_Energy_Consumed",
        "Back-up_Heater_Energy_Consumed",
        "Immersion_Heater_Energy_Consumed",
        "Boiler_Energy_Output",
    ],
}


def set_file_format(file_format="parquet"):
    """Specify file format for data input

    Args:
        file_format (str, optional): the file format, one of parquet or csv

    Raises:
        ValueError: if file format provided is not parquet or csv
    """
    file_formats = ["parquet", "csv"]
    if file_format not in file_formats:
        raise ValueError("Expected one of: parquet, csv")

    return file_format


def select_sensors(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Selects the columns required to calculate spfh2, spfh3 or spfh4

    Args:
        data (pd.DataFrame): dataframe with raw data on home
        metric (str): the variant of spfh to be calculated. Accepted values: spfh2, spfh3, spfh4

    Returns:
        pd.DataFrame: dataframe with only the columns required for the spfhx calculation.
                    - NB: The select_window function will behave differently depending on which columns are selected.
    """
    spfh_sensors = data.loc[data["sensor_type"].isin(spfh_required_cols[metric])].copy()

    return spfh_sensors


def round_timestamps(data: pd.DataFrame, n_mins: int = 2) -> pd.DataFrame:
    """Rounds timestamp with algorithm

    Args:
        spfh_sensors (pd.DataFrame): dataframe with data on sensors where index is the timestamp in datetime format
        n_mins (int, optional): Number of minutes to round to. Defaults to 2.

    Returns:
        pd.DataFrame: dataframe with rounded timestamps
    """

    data["Timestamp_rounded"] = data.index.round(f"{n_mins}T")

    # Code to correct timestamps
    # In case there happen to be duplicated timestamps we reset the index first
    data = data.reset_index()
    # Groupby makes sure that we are comparing the sensor with itself only
    data["Timestamp_rounded_diff_next"] = data.groupby("sensor_type")["Timestamp_rounded"].diff(
        periods=-1
    ).abs() / datetime.timedelta(minutes=1)
    data["Timestamp_rounded_diff_prev"] = data.groupby("sensor_type")["Timestamp_rounded"].diff(
        periods=1
    ).abs() / datetime.timedelta(minutes=1)
    correction_flag_mask = (data["Timestamp_rounded_diff_next"] == 0) & (
        data["Timestamp_rounded_diff_prev"] >= (n_mins * 2)
    )
    data["correction_flag"] = False
    data.loc[correction_flag_mask, "correction_flag"] = True
    data.loc[correction_flag_mask, "Timestamp_rounded"] = data.loc[
        correction_flag_mask, "Timestamp_rounded"
    ] - datetime.timedelta(minutes=n_mins)
    data.loc[(data["correction_flag"])]

    data = data.drop_duplicates(["sensor_type", "Timestamp_rounded"], keep="last")

    data = data.set_index("Timestamp_rounded")
    data.index.rename("Timestamp", inplace=True)
    data = data.drop(
        [
            "Timestamp_rounded_diff_next",
            "Timestamp_rounded_diff_prev",
            "correction_flag",
            "Timestamp",
        ],
        axis=1,
    )

    return data


def select_window(
    data: pd.DataFrame,
    gap_len_defs: dict,
    home_summary_part: pd.DataFrame,
    spf_ranges: dict,
    window_len_mths: int = 12,
    method: str = "last",
    plot_full_save_path: str = "",
    plot_window_save_path: str = "",
    location_out_cleaned: str = "",
    file: str = "",
    save_scored_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find find a window in the data with the required duration

    Args:
        data (pd.DataFrame): The data to find a window in
        gap_len_defs (dict): Definitions for different gap lengths
        home_summary_part (pd.DataFrame): DataFrame summarising outputs for this home, will be added to and output
        spf_ranges (dict): Acceptable SPF max and min values for different durations
        window_len_mths (int, optional): The required duration of the window. Defaults to 12.
        method (str, optional): Method to use to select the window, either "last" or "best". Defaults to "last".
        plot_full_save_path (str, optional): Path at which to save the plot of the full data. Use "none" to show figure. Defaults to "".
        plot_window_save_path (str, optional): Path at which to save the plot of the window data. Use "none" to show figure. Defaults to "".

    Returns:
        pd.DataFrame: The data for the selected window
        pd.DataFrame: home_summary_part modified with summary of window info
        pd.DataFrame: All the data after cleaning
        pd.DataFrame: All the available windows in the data
        pd.DataFrame: Record of alterations made to the data during cleaning
    """

    data_drop_nan = data.dropna()

    # If dropna results in no data left, we want to return
    if len(data_drop_nan) == 0:
        return data, home_summary_part, data, pd.DataFrame(), pd.DataFrame()

    if method == "last":
        end_spfh_timestamp = data_drop_nan.index.max() + pd.DateOffset(minutes=-2)
        max_score = np.nan
        mean_score = np.nan
    elif method == "best":
        (best_window, home_summary_part, windows, alteration_record, cleaned_data,) = ds.find_best_window(
            data_drop_nan,
            gap_len_defs,
            home_summary_part,
            spf_ranges,
            window_len_mths=window_len_mths,
            plot_full_save_path=plot_full_save_path,
            plot_window_save_path=plot_window_save_path,
            location_out_cleaned=location_out_cleaned,
            file=file,
            save_scored_data=save_scored_data,
        )

    if len(best_window) > 0:
        # filter dataframe within analysis window
        window_data = cleaned_data.loc[
            (cleaned_data.index <= best_window["end"].values[0])
            & (cleaned_data.index >= best_window["start"].values[0])
        ].copy()

        # Add the best window information to the output
        best_window = best_window.add_prefix("window_")
        best_window["Property_ID"] = home_summary_part["Property_ID"].values[0]
        home_summary_part = pd.merge(home_summary_part, best_window, on="Property_ID", how="outer")
        # spfh numbers don't have the prefix windows (This is how it was previously and we don't want to break analysis code down stream)
        home_summary_part = home_summary_part.rename(
            columns={
                "window_spfh2": "spfh2",
                "window_spfh3": "spfh3",
                "window_spfh4": "spfh4",
            }
        )

        home_summary_part["window_duration_days"] = pd.to_timedelta(
            home_summary_part["window_end"] - home_summary_part["window_start"]
        ) / pd.Timedelta(days=1)
        # If no window is found don't want to calculates the following
        if home_summary_part["window_duration_days"].values[0] > (pd.Timedelta(minutes=2) / pd.Timedelta(days=1)):
            expected_data_points = math.floor(
                home_summary_part["window_duration_days"] * pd.Timedelta(days=1) / datetime.timedelta(minutes=2)
            )
            for sensor in window_data["sensor_type"].unique():
                home_summary_part[f"window_%_complete_{sensor}"] = (
                    window_data.loc[window_data["sensor_type"] == sensor].dropna().count().value / expected_data_points
                )

            # We want to give the window a bad gap score if it has < 50% of its HP data
            HP_data_mask = (
                (home_summary_part["window_%_complete_Whole_System_Energy_Consumed"] < 0.5)
                | (home_summary_part["window_%_complete_Heat_Pump_Energy_Output"] < 0.5)
            ) & (home_summary_part["window_max_gap_score"] < 4)
            home_summary_part.loc[HP_data_mask, "window_max_gap_score"] = 4
            # Also need to update the max score which depends on the max gap score
            home_summary_part.loc[HP_data_mask, "window_max_score"] = home_summary_part.loc[
                HP_data_mask, ["window_max_gap_score", "window_max_data_score"]
            ].max(axis=1)

    else:
        window_data = pd.DataFrame()

    return window_data, home_summary_part, cleaned_data, windows, alteration_record


def calc_spf_for_all_windows(windows: pd.DataFrame) -> pd.DataFrame:

    spfh_metrics = ["spfh2", "spfh3", "spfh4"]
    # spf_data = pd.DataFrame()
    columns_present = windows.columns

    # Reset cumulative values to 0 of each sensor and find the last value
    for col in spfh_required_cols["spfh2"]:
        if col not in columns_present:
            if col in [
                "Back-up_Heater_Energy_Consumed",
                "Immersion_Heater_Energy_Consumed",
                "Circulation_Pump_Energy_Consumed",
            ]:
                # If certain columns are missing this isn't necessarily cause for concern
                # Back-up heater may never be used in many homes,
                # Home may not have an immersion heater
                # Some homes have had issues with data from circulation pumps
                windows[col] = 0
            else:
                # The other columns should be present and if missing and required in SPF calc then we want to return NaN
                windows[col] = np.NaN

    ## Populate the output DataFrame
    # Calculate the SPF for the given variant (metric)
    for metric in spfh_metrics:
        windows[metric] = spf(windows, metric)

    return windows


def spf(data: pd.DataFrame, metric: str) -> float:
    """Calculate a single Seasonal Performance Factor (SPF) metric given the total energies input and output for the period
    Logic for these calculations are not necessarily obvious. spfh2, 3 and 4 are metrics of performance for different parts of the heating system,
    each including different components of the whole system. This means the calculation for each metric must be different.
    SPFH2 includes the fewest components and SPFH4 the most.
    It is also important to understand that the parameter "Whole_System_Energy_Consumed" is the cumulative meter measuring the energy delivered to the
    whole system and not just the heat pump. To isolate the energy consumed by just the heat pump, one must therefore subtract the readings from all other components.
    This is done for the calculation of SPFH2 in this function.
    Another subtlety is that heat output from resistance heating components has not been metered and is assumed to be 100% efficient. It is for this reason that Immersion
    and Back-up Heater_Energy_Consumed readings are added to Heat_Pump_Energy_Output calculations for SPFH3 and 4.
    We also assume heat produced by the circulation pump to be negligible.

    Args:
        data (pd.DataFrame): Containing columns of the consumed and output energy.
        metric (str): The variant of SPFHX to be calculated. Accepted values: spfh2, spfh3, spfh4

    Returns:
        float: The calculated metric
    """

    if metric == "spfh2":
        numerator = data["Heat_Pump_Energy_Output"]
    elif (metric == "spfh3") or (metric == "spfh4"):
        numerator = (
            data["Heat_Pump_Energy_Output"]
            + data["Back-up_Heater_Energy_Consumed"]
            + data["Immersion_Heater_Energy_Consumed"]
        )

    if metric == "spfh2":
        denominator = (
            data["Whole_System_Energy_Consumed"]
            - data["Circulation_Pump_Energy_Consumed"]
            - data["Immersion_Heater_Energy_Consumed"]
            - data["Back-up_Heater_Energy_Consumed"]
        )
    elif metric == "spfh3":
        denominator = data["Whole_System_Energy_Consumed"] - data["Circulation_Pump_Energy_Consumed"]
    elif metric == "spfh4":
        denominator = data["Whole_System_Energy_Consumed"]

    spf_values = numerator / denominator

    # If the denominator is 0, we will get inf's in spf_values, we want these to be NaNs
    spf_values.loc[spf_values == np.inf] = np.NaN

    return spf_values


def add_time_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Adds date, hour and half hour fields to the dataframe, using indexed timestamp

    Args:
        data (pd.DataFrame): dataframe with the timestamp as index

    Returns:
        pd.DataFrame: dataframe with new time columns added
    """

    data["date"] = data.index.date
    data["hour"] = data.index.hour
    data["half_hour"] = np.where(
        data.index.minute < 30,
        data["hour"] * 2,
        data["hour"] * 2 + 1,
    )

    return data


def create_daily_plots(data: pd.DataFrame, metric: str, house_id: str, home_summary_part: pd.DataFrame):
    """Groups by day and creates graph if spfh value was calculated

    Args:
        data (pd.DataFrame): dataframe with data at 2 minute intervals
        metric (str): the spfh to pick the right columns for the graph
        house_id (str): the Property_ID
        home_summary_part (pd.DataFrame): to check if the spfh was calculated
    """
    if metric in home_summary_part.columns:
        spfh_sensors_daily = add_time_columns(data)
        # spfh_sensors_daily["date"] = pd.to_datetime(spfh_sensors_daily["date"])
        spfh_sensors_daily = spfh_sensors_daily.groupby("date").max()
        spfh_sensors_daily = spfh_sensors_daily.drop(["hour", "half_hour"], axis=1)
        spfh_sensors_daily = spfh_sensors_daily.dropna()

        fig = plt.rcParams["figure.figsize"] = (20, 10)
        plt.title(house_id)

        for col in spfh_required_cols[metric]:
            if col in spfh_sensors_daily.columns:
                spfh_sensors_daily[col] = spfh_sensors_daily[col] - spfh_sensors_daily[col].iloc[0]
                fig = plt.plot(spfh_sensors_daily.index, spfh_sensors_daily[col], label=col)

        plt.legend(loc="upper left")
        plt.xlabel("")
        plt.savefig(os.path.join("plots", metric, house_id))
        plt.close()


def create_home_summary(data: pd.DataFrame, id: str) -> pd.DataFrame:
    """Creates the output dataframe for the spfh analysis to be populated

    Args:
        data (pd.DataFrame): full data for a home
        id (str): site reference id

    Returns:
        pd.DataFrame: dataframe for the spfh analysis to be populated
    """
    Property_ID = id
    home_summary_part = pd.DataFrame([Property_ID], columns=["Property_ID"])

    # For the start and end times and the completeness %'s we want to know these for data that aren't NaN
    data = data.dropna()

    # Start and end times are only helpful if they have both HP_energy_consumed and HP_energy_output readings
    # So here we filter to find the useful timestamps where we have both of these
    hp_con = data[data["sensor_type"] == "Whole_System_Energy_Consumed"].index
    hp_out = data[data["sensor_type"] == "Heat_Pump_Energy_Output"].index
    useful_index = hp_con.intersection(hp_out)

    home_summary_part["start"] = useful_index.min()
    home_summary_part["end"] = useful_index.max()
    home_summary_part["duration_days"] = pd.to_timedelta(
        home_summary_part["end"] - home_summary_part["start"]
    ) / pd.Timedelta(days=1)
    if not home_summary_part["duration_days"].isna().values[0]:
        expected_data_points = math.floor(
            home_summary_part["duration_days"] * pd.Timedelta(days=1) / pd.Timedelta(minutes=2)
        )

        sensor_timestamp_count = (
            data[["sensor_type", "value"]].groupby(by="sensor_type").count().T / expected_data_points
        )
        sensor_timestamp_count = sensor_timestamp_count.add_prefix("%_complete_")
        home_summary_part = pd.merge(home_summary_part, sensor_timestamp_count, how="cross")

    return home_summary_part


def further_qa(data: pd.DataFrame, metric: str, home_summary_part: pd.DataFrame) -> pd.DataFrame:
    possible_timestamps_1_yr = 262800
    home_summary_part[f"{metric}_missing"] = 1 - len(data) / possible_timestamps_1_yr
    home_summary_part[f"{metric}_decreasing"] = (
        sum(((data["Whole_System_Energy_Consumed"].diff(-1)) > 0)) / possible_timestamps_1_yr
    )
    return home_summary_part


def remove_flat_start(spfh_sensors: pd.DataFrame, col: str) -> int:
    """If a column starts with a constant value for a period of time, set that whole flat period to nan
    (on the assumption the meter hasn't started working yet)

    Args:
        spfh_sensors (pd.DataFrame): sensor data
        col (str): column of interest

    Returns:
        num_points_reset (int): number of points set to nan
    """
    # Flag whether this row has the same value as the previous row.
    # Use >0 rather than !=0 so nan's return False
    spfh_sensors[f"{col}_diff"] = spfh_sensors[col].diff(1) > 0
    # Until the cumulative sum of that exceeds zero then we have a flat line, so set that flat period to nan
    spfh_sensors[f"{col}_diff_sum"] = spfh_sensors[f"{col}_diff"].cumsum()
    spfh_sensors.loc[spfh_sensors[f"{col}_diff_sum"] == 0, col] = np.nan

    num_points_reset = (spfh_sensors[f"{col}_diff_sum"] == 0).sum()
    spfh_sensors.drop(columns=[f"{col}_diff", f"{col}_diff_sum"], inplace=True)

    return num_points_reset


def flag_flat(spfh_sensors: pd.DataFrame, col: str, duration: int = 15):
    """Flags whether a series is flat for a particular duration going forwards

    Args:
        spfh_sensors (pd.DataFrame): sensor data
        col (str): column of interest
        duration (int, optional): How many intervals to look forward. Defaults to 15 (i.e. 30 mins).
    """
    # We forward fill so that nans tend to count as flat, and use -duration so we are comparing to
    # the row n to row n+duration (rather than row n to row n-duration)
    spfh_sensors[f"{col}_flat_{duration*2}_mins"] = spfh_sensors[col].ffill().diff(-duration) == 0


def select_sensors(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Selects the columns required to calculate spfh2, spfh3 or spfh4

    Args:
        data (pd.DataFrame): dataframe with raw data on home
        metric (str): the variant of spfh to be calculated. Accepted values: spfh2, spfh3, spfh4

    Returns:
        pd.DataFrame: dataframe with only the columns required for the spfhx calculation.
                    - NB: The select_window function will behave differently depending on which columns are selected.
    """
    spfh_sensors = data.loc[data["sensor_type"].isin(spfh_required_cols[metric])].copy()

    return spfh_sensors


def Property_ID_home_id_lookup() -> pd.DataFrame:
    """Returns a full lookup of Property_ID and home_id directly from usmart

    Returns:
        pd.DataFrame: a lookup of Property_ID and home_id
    """

    pre_ = "https://api.usmart.io/org"
    headers = {
        "cache-control": "no-cache",
        "api-key-id": os.environ["MEP_KEY_ID"],
        "api-key-secret": os.environ["MEP_KEY_SECRET"],
    }

    urls = {
        "ovo": [
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/6aaee576-c76f-4974-90c2-0f379782df7d",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/b808ba0e-2d9a-4da6-8bc0-6630ca6e1eab",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/f4e01536-eded-473e-9022-69d889d8b18d",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/836a931a-bb2c-4b87-96a1-553fffd7de62",
        ],
        "eon": [
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/4e44ea07-a65b-4382-b888-45651fc09901",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/84809e09-33dc-48b5-a4c3-90cc7efe00be",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/bbffa51b-9867-4cad-9056-9b589123ca37",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/7709a77c-c7d5-4121-8e46-541e2600e59f",
        ],
        "ww": [
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/3309c112-07bc-4fe8-986c-4e52861eb1b8",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/59d17bb5-e289-442f-b778-f26dc786ffcb",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/5f12eaf2-b998-420b-9555-31e26b783314",
            "92610836-6c2b-4a26-a0a0-b903bde0dc46/4dd7a850-858c-4e65-a456-772f76e93ec1",
        ],
    }

    try:
        # Takes ~30 secs
        installers = ["ovo", "eon", "ww"]
        dfs = []
        for installer in installers:
            # print(installer)
            for url_in in urls[installer]:
                # print(url_in)
                url = (
                    "https://api.usmart.io/org/"
                    + url_in
                    + "/latest/urql?"
                    + "&aggregate(Property_ID,Home_ID,value_count(Timestamp))"
                )
                response = requests.request("GET", url, headers=headers)
                file = json.loads(response.content)[0]
                df = pd.DataFrame(file).T

                if len(df) == 0:
                    continue
                dfs.append(df)

                df = pd.concat(dfs)
                df = df.reset_index()
                df = pd.melt(df, id_vars="index")
                df = df.dropna()
                df = df.drop("value", axis=1)
                # df.to_csv("missing_data_summary.csv", index=False)
            # If we hit an error it's usually due to a problem with the data coming back, so print that out
        df.T
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("OOps: Something Else", err)

    df = df.groupby(["index", "variable"]).count().reset_index()
    df = df.rename({"index": "Property_ID", "variable": "Home_ID"}, axis=1)
    # df.to_csv("Property_ID-Home_ID Lookup.csv", index=False)

    return df


def get_installation_database() -> pd.DataFrame:
    """Get the installation database records from USMART

    Returns:
        pd.DataFrame: Installation database
    """
    urls = [
        "https://api.usmart.io/org/92610836-6c2b-4a26-a0a0-b903bde0dc46/36f81d29-023d-493c-b762-c21096e1a2c7/latest/urql",
        "https://api.usmart.io/org/92610836-6c2b-4a26-a0a0-b903bde0dc46/6aa8fedc-406b-4eb2-a7b3-fcd77170e867/latest/urql",
        "https://api.usmart.io/org/92610836-6c2b-4a26-a0a0-b903bde0dc46/6cdd1c79-167f-48fa-bae4-2c8ab659886a/latest/urql",
    ]
    headers = {
        "cache-control": "no-cache",
        "api-key-id": os.environ["MEP_KEY_ID"],
        "api-key-secret": os.environ["MEP_KEY_SECRET"],
    }

    installation_database = []
    for url in urls:
        response = requests.request("GET", url, headers=headers)
        file = pd.DataFrame(json.loads(response.text))
        installation_database.append(file)

    installation_database = pd.concat(installation_database)
    installation_database = installation_database[
        [
            "House_ID",
            "HP_Installed",
            "HP_Size_kW",
            "HP_Brand",
            "HP_Model",
            "Name_Install",
        ]
    ]
    anonymised_ids = pd.read_excel(
        "S:\Projects\Electrification of Heat\WP3 - Data Collection & Co-ordination\Property Number Anonymisation.xlsx"
    )
    installation_database = installation_database.merge(anonymised_ids, how="inner", on="House_ID")
    hp_simplify_dict = {
        "ASHP": "ASHP",
        "Hybrid_Split_New": "Hybrid",
        "HT_ASHP": "HT_ASHP",
        "GSHP_Borehole": "GSHP",
        "Hybrid_Monobloc": "Hybrid",
        "GSHP": "GSHP",
        "Hybrid": "Hybrid",
        "Hybrid_split_new": "Hybrid",
        "Hybrid_Split_Existing": "Hybrid",
    }
    installation_database["HP_Type"] = installation_database["HP_Installed"].map(hp_simplify_dict)
    installation_database["DC"] = np.where(
        installation_database["House_ID"].astype(str).str.len() == 7,
        "OVO",
        np.where(
            installation_database["House_ID"].astype(str).str.len() == 6,
            "E.ON",
            "Warmworks",
        ),
    )
    return installation_database
