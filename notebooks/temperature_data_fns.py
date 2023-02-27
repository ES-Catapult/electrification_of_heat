from turtle import home
import pandas as pd
import plotly.express as px
import plotly.io as pio
from numpy import float64
import numpy as np
import utils
import qa_functions as qa
import plotly.graph_objects as go
import os
import data_scoring_fns as ds


def plot_data(data: pd.DataFrame, path: str = "", freq: str = "30T", agg_fn: str = "mean"):
    """Plot the data at the given frequency

    Args:
        data( pd.DataFrame): The data to plot
        path (str, optional): The path to save the plot. Defaults to "".
        freq (str, optional): The frequency at which to plot the data. Defaults to "30T".
    """
    if "Timestamp" in data.columns:
        data = data.set_index("Timestamp")

    plot_data = data.groupby("sensor_type")["value"].resample(freq).agg(agg_fn).reset_index()
    fig = px.scatter(plot_data, x="Timestamp", y="value", color="sensor_type")
    fig["layout"].update(height=400, width=1000, margin=dict(l=50, r=20, t=40, b=10))

    if path != "":
        pio.write_image(fig, path, format="png", engine="orca")
    else:
        fig.show()


def get_temperature_stats(data: pd.DataFrame, id: str) -> pd.DataFrame:
    """Calculate and return certain statistics of the given temperature data

    Args:
        data (pd.DataFrame): The temperature data to be analysed
        id (str): ID to label the data with

    Returns:
        pd.DataFrame: The calculated statistics
    """

    data["diff"] = data.groupby("sensor_type")["value"].diff()

    stats_temp = data.groupby("sensor_type")[["value", "diff"]].agg(["mean", "std"]).reset_index()

    stats_data = pd.DataFrame(stats_temp["sensor_type"])
    stats_data["Property_ID"] = id
    for col0 in ["value", "diff"]:
        for col1 in ["mean", "std"]:
            stats_data[f"{col1}: {col0}"] = stats_temp[col0][col1]

    stats_data["mean: daily max"] = (
        data.set_index("Timestamp")
        .groupby("sensor_type")
        .resample("1d")
        .agg({"value": "max"})
        .reset_index()
        .groupby("sensor_type")["value"]
        .mean()
        .reset_index()["value"]
    )
    stats_data["mean: daily min"] = (
        data.set_index("Timestamp")
        .groupby("sensor_type")
        .resample("1d")
        .agg({"value": "min"})
        .reset_index()
        .groupby("sensor_type")["value"]
        .mean()
        .reset_index()["value"]
    )

    for sensor in stats_data["sensor_type"].unique():
        stats_data.loc[stats_data["sensor_type"] == sensor, "spikiness: value"] = spikiness(
            data.loc[data["sensor_type"] == sensor, "value"]
        )
        stats_data.loc[stats_data["sensor_type"] == sensor, "spikiness: diff"] = spikiness(
            data.loc[data["sensor_type"] == sensor, "diff"]
        )

    stats_data["mean: count per day"] = (
        data.dropna(subset="diff")
        .set_index("Timestamp")
        .groupby("sensor_type")
        .resample("1d")["value"]
        .count()
        .dropna()
        .reset_index()
        .groupby("sensor_type")["value"]
        .mean()
        .reset_index()["value"]
    )

    return stats_data


def spikiness(data: pd.Series) -> float64:
    """Calculate spikiness for a pandas data Series.
    The basis of this calculation is the spikiness function:
    int( (y''(t))**2 )dt - The integral of the square of the second derivative of the function
    Here we calculate the RMS spikiness for evenly spaced data
    This function is copied from the ESC github: https://github.com/esc-data-science/spikiness/blob/main/spikiness.py
    Args:
        data (pd.Series): The pandas Series for which spikiness will be evaluated
    Returns:
        float64: The spikiness calculated for data
    """
    # Normalize the data so we calculate spikiness only (not influenced by size of values in data)
    y = data / data.abs().mean()

    y_dd = y.diff().diff()
    y_dd_sqrd = y_dd**2

    # Use fillna(0) so that mean is taken over the full number of original points
    spikiness = np.sqrt(y_dd_sqrd.fillna(0).mean())

    return spikiness


def remove_temperature_anomalies(
    data: pd.DataFrame, normal_ranges: dict, alteration_record: pd.DataFrame
) -> pd.DataFrame:

    temperature_data = utils.filter_temperature_data(data)
    temperature_data = temperature_data.reset_index()

    # Flag values outside of their specified range
    temperature_data.loc[:, "max"] = temperature_data["sensor_type"].map(lambda x: normal_ranges[x]["max"])
    temperature_data.loc[:, "min"] = temperature_data["sensor_type"].map(lambda x: normal_ranges[x]["min"])
    temperature_data["anomalous"] = ~(
        (temperature_data["value"] <= temperature_data["max"]) & (temperature_data["value"] >= temperature_data["min"])
    )

    # Update the alteration record with the anomalies removed
    if temperature_data["anomalous"].sum() > 0:
        anomalous_data = temperature_data.loc[temperature_data["anomalous"]]
        for sensor in anomalous_data["sensor_type"].unique():
            senor_data = anomalous_data.loc[anomalous_data["sensor_type"] == sensor]
            alteration_record = ds.add_alteration_record(
                alteration_record,
                changed_data=senor_data,
                alteration=f"""{senor_data["anomalous"].sum()} data point(s) removed""",
                reason="Anomalous point(s)",
            )

    # Remove the anomalous data
    temperature_data = temperature_data.loc[
        temperature_data["anomalous"] == False, ["Timestamp", "sensor_type", "value"]
    ]

    # Replace un-cleaned temperature data from data with cleaned temperature data
    if "Timestamp" not in data.columns:
        data = data.reset_index()
    non_temperature_data = data.loc[~data["sensor_type"].isin(temperature_data["sensor_type"].unique())]
    cleaned_data = pd.concat([non_temperature_data, temperature_data])

    return cleaned_data, alteration_record


def calculate_heating_temp_averages(data: pd.DataFrame, file_path: str = "", plot_path: str = "") -> pd.DataFrame:

    heating_temp_sensors = ["Heat_Pump_Heating_Flow_Temperature", "Heat_Pump_Return_Temperature"]
    # Keep only the relevant data, within the time window
    data = qa.round_timestamps(data, n_mins=2)
    data = data.pivot(columns="sensor_type", values="value")
    data = data.reset_index()
    # We only want to temperatures measured while the heat pump was on
    if "Heat_Pump_Energy_Output" in data.columns:
        heat_pump_on_mask = data["Heat_Pump_Energy_Output"].diff() > 0
        data = data.loc[heat_pump_on_mask]
    # We only want temperatures when output was used for heating, not hot water
    if "Hot_Water_Flow_Temperature" in data.columns:
        heating_on_mask = data["Hot_Water_Flow_Temperature"].isna()
        data = data.loc[heating_on_mask]

    # If the heating temperature sensors are present then we should continue, otherwise we can't find their averages
    if (heating_temp_sensors[0] in data.columns) & (heating_temp_sensors[1] in data.columns):
        data = data[["Timestamp"] + heating_temp_sensors]
    else:
        return pd.DataFrame(
            {heating_temp_sensors[0]: [np.nan, np.nan, np.nan], heating_temp_sensors[1]: [np.nan, np.nan, np.nan]},
            index=["mean", "median", "mode"],
        )

    averages = data[heating_temp_sensors].agg(["mean", "median"])

    # Bin the temperatures within a reasonable range for the heating flow temperature
    start = 20
    end = 75
    step = 0.5
    bins = np.arange(start - (step / 2), end + (step / 2), step)
    labels = np.arange(start, end, step)
    data = data.melt(id_vars="Timestamp")
    data["bin"] = pd.cut(x=data["value"], bins=bins, labels=labels)
    binned_data = data.groupby(["sensor_type", "bin"])["value"].count()
    if file_path != "":
        binned_data.to_csv(file_path)
    max_counts = binned_data.reset_index(0).groupby("sensor_type").max()

    # Find the most common flow temperature
    for sensor in heating_temp_sensors:
        averages.loc["mode", sensor] = (
            binned_data.loc[sensor].loc[binned_data.loc[sensor] == max_counts.loc[sensor].values[0]].index[0]
        )

    if plot_path == "none":
        return averages
    else:
        plotting_data = binned_data.reset_index()
        # Plot a the binned data with mean median and mode lines
        fig = px.scatter(plotting_data, x="bin", y="value", color="sensor_type")
        max_count = plotting_data["value"].max()
        for ii, sensor in enumerate(heating_temp_sensors):
            mean = averages.loc["mean", sensor]
            median = averages.loc["median", sensor]
            mode = averages.loc["mode", sensor]
            color = fig.data[ii].marker.color
            fig.add_trace(
                go.Scatter(
                    x=[mean, mean],
                    y=[0, max_count],
                    mode="lines",
                    name=f"mean: {round(mean,1)}",
                    line_color=color,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[median, median],
                    y=[0, max_count],
                    mode="lines",
                    line_dash="dash",
                    name=f"median: {round(median,1)}",
                    line_color=color,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[mode, mode],
                    y=[0, max_count],
                    mode="lines",
                    line_dash="dot",
                    name=f"mode: {round(mode,1)}",
                    line_color=color,
                )
            )
        fig["layout"].update(
            height=400,
            width=1000,
            margin=dict(l=50, r=20, t=40, b=10),
            yaxis={"title": "count"},
            xaxis={"title": "Heating temperature"},
        )

        if plot_path == "":
            fig.show()
        else:
            pio.write_image(fig, plot_path, format="png", engine="orca")

    return averages


def calculate_high_temperature_stats(data: pd.DataFrame) -> pd.DataFrame:

    # We want to calculate theses stats for times when the heat pump is running
    sensors_to_keep = [
        "Heat_Pump_Heating_Flow_Temperature",
        "Heat_Pump_Hot_Water_Flow_Temperature",
        "Heat_Pump_Energy_Output",
    ]
    data = data[data["sensor_type"].isin(sensors_to_keep)]
    data = data.reset_index()
    data = data.pivot(index="Timestamp", columns="sensor_type", values="value")
    data = data.loc[data["Heat_Pump_Energy_Output"].diff() > 0]

    data = data.reset_index().melt(id_vars="Timestamp")
    flow_temps = ["Heat_Pump_Heating_Flow_Temperature", "Heat_Pump_Hot_Water_Flow_Temperature"]
    data = data.loc[data["sensor_type"].isin(flow_temps)].dropna()

    stats = dict()
    stats["flow > 65C"] = (data["value"] > 65).any()
    if data["value"].count() > 0:
        stats["% > 65C"] = (data["value"] > 65).sum() / data["value"].count()
    else:
        stats["% > 65C"] = np.nan

    return stats


def add_flow_temp_stats_for_window(
    data: pd.DataFrame, home_summary: pd.DataFrame, id: str, file_path: str = "", plot_path: str = ""
) -> pd.DataFrame:

    # Get the start and end times for the window
    id_mask = home_summary["Property_ID"].astype(str) == str(id)
    start = pd.to_datetime(home_summary.loc[id_mask, "window_start"].values[0])
    end = pd.to_datetime(home_summary.loc[id_mask, "window_end"].values[0])

    # If the start time doesn't exist, this means no window was found for this home, so just return the home_summary and stop
    if not ((end - start) == pd.Timedelta(days=365)):
        return home_summary
    else:
        data = data.reset_index()
        data = data[(data["Timestamp"] >= start) & (data["Timestamp"] <= end)]
        data = data.set_index("Timestamp")

    # Add the heating flow temperature stats for the data
    averages = calculate_heating_temp_averages(data.copy(), file_path=file_path, plot_path=plot_path)
    # Add the temperatures to the home output
    home_summary = add_temp_stats_to_output(home_summary, averages, id)

    # We also want to add flow temperature stats for the HP as a whole
    # So we will include both the heating flow temp and hot water flow temp
    flow_temp_data = data.copy()
    flow_temp_data.loc[
        flow_temp_data["sensor_type"] == "Hot_Water_Flow_Temperature", "sensor_type"
    ] = "Heat_Pump_Heating_Flow_Temperature"
    file_path = os.path.join(os.path.split(file_path)[0], "all_flow_temperatures", os.path.split(file_path)[1])
    plot_path = os.path.join(os.path.split(plot_path)[0], "all_flow_temperatures", os.path.split(plot_path)[1])
    averages = calculate_heating_temp_averages(flow_temp_data.copy(), file_path=file_path, plot_path=plot_path)
    averages = averages.drop(columns=["Heat_Pump_Return_Temperature"])
    averages = averages.rename(columns={"Heat_Pump_Heating_Flow_Temperature": "HP_HW_Flow_Temperature"})
    # Add the temperatures to the home output
    home_summary = add_temp_stats_to_output(home_summary, averages, id)

    HT_stats = calculate_high_temperature_stats(data.copy())
    for stat in HT_stats.keys():
        home_summary.loc[id_mask, f"window_{stat}"] = HT_stats[stat]

    return home_summary


def add_temp_stats_to_output(home_summary: pd.DataFrame, averages: pd.DataFrame, id: str) -> pd.DataFrame:
    # Add the temperature to the home output
    id_mask = home_summary["Property_ID"].astype(str) == str(id)
    column_list = averages.columns
    metric_list = averages.index
    for metric in metric_list:
        for col in column_list:
            home_summary.loc[id_mask, f"window_{metric}_{col}"] = averages.loc[metric, col]
        home_summary.loc[id_mask, f"window_flow_return_{metric}_flagged"] = (
            home_summary.loc[id_mask, f"window_{metric}_Heat_Pump_Heating_Flow_Temperature"]
            - home_summary.loc[id_mask, f"window_{metric}_Heat_Pump_Return_Temperature"]
        )

    # HT_stats = calculate_high_temperature_stats(data.copy())
    # for stat in HT_stats.keys():
    #     home_summary.loc[id_mask, f"window_{stat}"] = HT_stats[stat]

    return home_summary


def add_spfs_for_coldest_period(
    data: pd.DataFrame, id: str, duration: pd.Timedelta, prefix_label: str
) -> pd.DataFrame:
    """Calculate the SPFs for the coldest period of specified duration

    Args:
        data (pd.DataFrame): The data to calculate the SPFs for
        id (str): The ID of the data
        duration (pd.Timedelta): Duration of the required period
        prefix_label (str): The label to prefix the output columns with

    Returns:
        pd.DataFrame: The output data frame
    """

    if ("Heat_Pump_Energy_Output" in data["sensor_type"].unique()) & (
        "External_Air_Temperature" in data["sensor_type"].unique()
    ):

        # Find the datetime of the coldest period (with at least some HP usage)
        data_part = data.loc[
            data["sensor_type"].isin(["External_Air_Temperature", "Heat_Pump_Energy_Output"])
        ].reset_index()
        data_part = data_part.pivot(index="Timestamp", columns="sensor_type", values="value")
        data_part = (
            data_part.resample(duration)
            .agg({"External_Air_Temperature": "mean", "Heat_Pump_Energy_Output": "max"})
            .reset_index()
        )
        data_part = data_part.loc[data_part["Heat_Pump_Energy_Output"].diff() > 0]
        if len(data_part) > 0:
            start_time = data_part.loc[
                data_part["External_Air_Temperature"] == data_part["External_Air_Temperature"].min(), "Timestamp"
            ].values[0]
        else:
            # If after all the filters there is no data left, return an empty data frame
            return pd.DataFrame({"Property_ID": id}, index=[0])

        # Round the data to nearest 2 minutes
        data = qa.round_timestamps(data, n_mins=2)

        # Trim all the data to the coldest period
        end_time = start_time + duration
        mask = (data.index >= start_time) & (data.index < end_time)
        data = data.loc[mask]

        # Get internal and external temperature data description
        temp_stats = (
            data[data["sensor_type"].isin(["External_Air_Temperature", "Internal_Air_Temperature"])]
            .groupby("sensor_type")
            .describe()
        )
        temp_stats["dummy_index"] = 0
        temp_stats = temp_stats.reset_index().pivot(index="dummy_index", columns="sensor_type")
        col_names = [f"{multi_ind[2]}: {multi_ind[1]}" for multi_ind in temp_stats.columns]
        temp_stats = temp_stats.droplevel(0, axis=1).droplevel(0, axis=1)
        temp_stats.columns = col_names
        temp_stats = temp_stats.sort_index(axis=1)

        # Keep only the energy meter data
        data = qa.select_sensors(data, metric="spfh2")

        # If there is any data in the period then we want to continue
        if len(data) > 0:
            # Round the time stamps
            data = qa.round_timestamps(data, n_mins=2)

            # Create standard output for the period
            home_summary_part = qa.create_home_summary(data, id)

            # Add the temperature descriptive stats to the output
            home_summary_part = pd.concat([home_summary_part, temp_stats], axis=1)

            # Get the data in the expected format
            data = data.pivot(columns=["sensor_type"], values=["value"])
            data.columns = data.columns.droplevel(level=0)

            # Calculate the SPFs
            window = pd.DataFrame(data.iloc[-1] - data.iloc[0]).T
            window = qa.calc_spf_for_all_windows(window)
            window["Property_ID"] = home_summary_part["Property_ID"]
            home_summary_part = pd.merge(home_summary_part, window, on="Property_ID")

            # Prefix the new columns and merge them into the output dataframe
            home_summary_part = home_summary_part.add_prefix(prefix_label)
            home_summary_part = home_summary_part.rename(columns={f"{prefix_label}Property_ID": "Property_ID"})

            return home_summary_part

        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
