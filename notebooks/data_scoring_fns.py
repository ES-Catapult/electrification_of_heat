import pandas as pd
import qa_functions as qa
import numpy as np
from typing import Tuple
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


def add_alteration_record(
    alteration_record: pd.DataFrame, changed_data: pd.DataFrame, alteration: str, reason: str,
) -> pd.DataFrame:
    """Add a record of an alteration to the alteration record

    Args:
        alteration_record (pd.DataFrame): Record of all the alterations made in this run
        changed_data (pd.DataFrame): The data being changed in this alteration
        alteration (str): The alteration made to the data
        reason (str): The reason for the alteration

    Returns:
        pd.DataFrame: The updated record of alterations
    """
    if type(changed_data) == pd.Series:
        changed_data = pd.DataFrame(changed_data).T
    sensor_type = changed_data["sensor_type"].unique()

    if "Timestamp" in changed_data.columns:
        start = changed_data["Timestamp"].min()
        end = changed_data["Timestamp"].max()
    else:
        start = changed_data.index.min()
        end = changed_data.index.max()

    new_alteration = {
        "sensor_type": sensor_type,
        "start_time": start,
        "end_time": end,
        "alteration": alteration,
        "reason": reason,
    }

    alteration_record = pd.concat([alteration_record, pd.DataFrame(new_alteration)])

    return alteration_record


def resample_data(data: pd.DataFrame, freq: str = "30T", agg_fn: str = "max") -> pd.DataFrame:
    """Resample cumulative data to specified frequency (.resample is slow)

    Args:
        data (pd.DataFrame): The data to resample
        freq (str): The frequency to resample the data at. Defaults to "30T".
        agg_fn (str): The function used to aggregate the data Defaults to "max".

    Returns:
        pd.DataFrame: The resampled data
    """
    data = data.groupby(["sensor_type", pd.Grouper(freq=freq)]).agg(agg_fn).reset_index()

    return data


def load_data(path: str, spfhx: str = "spfh2") -> pd.DataFrame:
    """Load the data from file and format it as expected

    Args:
        path (str): The path to the data to load
        spfhx (str): The spfh variables required

    Returns:
        pd.DataFrame: The data
    """
    raw_data = pd.read_parquet(path)
    raw_data.index = pd.to_datetime(raw_data.index)

    data = qa.select_sensors(raw_data, metric=spfhx)
    data = qa.round_timestamps(data, n_mins=2)

    return data


def fill_gaps_with_nans(data: pd.DataFrame) -> pd.DataFrame:
    """Fill the gaps in the data frame with NaNs. This is done by resampling at a 2 minute period

    Args:
        data (pd.DataFrame): The data to be resampled

    Returns:
        pd.DataFrame: The data once resampled with NaNs where values are missing
    """
    full_data = pd.DataFrame()
    sensor_types = data["sensor_type"].unique()
    for sensor in sensor_types:
        filt = data["sensor_type"] == sensor
        df = data[filt].apply(
            lambda x: x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="2T"), fill_value=np.NAN,)
        )
        df.index.set_names("Timestamp", inplace=True)
        df["sensor_type"] = df["sensor_type"].ffill()
        full_data = pd.concat([full_data, df])

    return full_data


def remove_start_with_spf_out_of_range(
    data: pd.DataFrame, spf_ranges: dict, alteration_record: pd.DataFrame
) -> pd.DataFrame:
    """Identify periods of the data with unusually low output from the heat pump.
        We suspect this data is incorrect. Remove periods like this from the start of the data.

    Args:
        data (pd.DataFrame): The data to analyse and process.
        spf_ranges (dict): Acceptable SPF max and min values for different durations
        alteration_record (pd.DataFrame): Record of alterations made to the data

    Returns:
        pd.DataFrame: The processed data
    """
    # We can only check if the output is low if we have HP energy consumed and output
    sensor_types = data["sensor_type"].unique()
    if ("Heat_Pump_Energy_Output" in sensor_types) & ("Whole_System_Energy_Consumed" in sensor_types):

        # Score all the data based daily spf values and if all of it is flat.
        daily_scored_data = score_all_data(data.copy().dropna(), spf_ranges, freq="d", duration="short")
        daily_scored_data = daily_scored_data.reset_index()

        # If the data has been scored 3.3 then it has an spf out of range or it is all flat.
        # These are days that we want to remove
        if (daily_scored_data["data_score"] != 3.3).any():
            # Scores not = 3.3 need to be kept
            new_start = daily_scored_data.loc[daily_scored_data["data_score"] != 3.3, "Timestamp"].min()
        else:
            # If no data_score is not 3.3 then new_start = the end of the data
            new_start = daily_scored_data["Timestamp"].max()

        remove_mask = data.index < new_start
        alteration_record = add_alteration_record(
            alteration_record,
            changed_data=data.loc[remove_mask],
            alteration="Data removed",
            reason="COP out of expected range",
        )
        # Replace the unwanted data with NaNs
        data.loc[remove_mask, "value"] = np.nan

        # Reset the level of the cumulative functions to start at 0 for their new first reading
        for sensor in sensor_types:
            filt = data["sensor_type"] == sensor
            df = data.loc[filt, "value"].dropna()
            if len(df) > 0:
                start_value = df.iloc[0]
                data.loc[filt, "value"] = data.loc[filt, "value"] - start_value

    # else:
    # If we don't have the required data, we don't want to change anything

    return data, alteration_record


def remove_anomalous_points(data: pd.DataFrame, alteration_record: pd.DataFrame) -> pd.DataFrame:
    """Find anomalous points and remove them

    Args:
        data (pd.DataFrame): The data
        alteration_record (pd.DataFrame): Record of alterations made to the data

    Returns:
        pd.DataFrame: The data with anomalies removed
    """
    # As the meter data is cumulative, to find anomalies we find the minimum of the previous 3 values
    # and the max of the following 3. All values outside of these metrics are flagged as anomalous
    data = data.reset_index()
    data["backward_rolling_min"] = (
        data.groupby("sensor_type").rolling(window=3)["value"].min().shift(periods=1).droplevel(0)
    )
    data["forward_rolling_max"] = (
        data.groupby("sensor_type").rolling(window=3)["value"].max().shift(periods=-3).droplevel(0)
    )
    # We know that there are some small decreases in some cumulative meters, we allow for this using the 0.95*backward_rolling_min
    data["anomalous"] = (data["value"] < 0.95 * data["backward_rolling_min"]) | (
        data["value"] > data["forward_rolling_max"]
    )

    # Anomalous values are removed and the change is recorded
    data.loc[data["anomalous"], "value"] = np.nan
    for sensor in data.loc[data["anomalous"], "sensor_type"].unique():
        changed_data = data.loc[data["anomalous"] & (data["sensor_type"] == sensor)]
        alteration_record = add_alteration_record(
            alteration_record,
            changed_data=changed_data,
            alteration=f"""{changed_data["anomalous"].sum()} data point(s) removed""",
            reason="Anomalous point(s)",
        )

    # Restore data to its original format for continued analysis after this function
    data = data.set_index("Timestamp")
    data = data.drop(columns=["backward_rolling_min", "forward_rolling_max", "anomalous"])

    return data, alteration_record


def find_gaps(data: pd.DataFrame, threshold: pd.Timedelta) -> pd.DataFrame:
    """Find the gaps in the data

    Args:
        data (pd.DataFrame): The data to find the gaps in
        threshold (pd.Timedelta): The duration over which data must be missing for us to classify it as a gap

    Returns:
        pd.DataFrame: DataFrame containing information on the gaps in the given data
    """
    sensor_types = data["sensor_type"].unique()
    data = data.reset_index()
    data_wide = data.pivot(index="Timestamp", columns="sensor_type", values="value")
    gaps_df = pd.DataFrame()

    # Identify the gaps
    for sensor in sensor_types:
        all_diff_list = []
        # For each sensor, find the difference in timestamps
        df = data.loc[data["sensor_type"] == sensor, :].copy()
        df["Gap duration"] = df["Timestamp"].diff()
        # Keep gaps with duration above the threshold (i.e. is the timestamp difference long enough to constitute a gap)
        df = df.loc[df["Gap duration"] > threshold]

        if not df.empty:
            # For each gap
            for ind in df.index:
                # Calculate and keep all delta values across the gap
                end_time = df.loc[ind, "Timestamp"]
                start_time = end_time - pd.to_timedelta(df.loc[ind, "Gap duration"])
                end_values = data_wide.loc[end_time].copy()
                start_values = data_wide.loc[start_time].copy()
                diff_values = end_values - start_values
                diff_values["index"] = ind

                all_diff_list.append(diff_values.copy())

            all_diffs = pd.concat(all_diff_list, axis=1)
            # If a sensor is not present, add its value as zero
            sensors_to_add = list(set(qa.spfh_required_cols["spfh2"]) - set(sensor_types))
            for column in sensors_to_add:
                all_diffs.loc[column] = 0

            if "index" in all_diffs.index:
                df = pd.concat([df, all_diffs.T.set_index("index")], axis=1)

            gaps_df = pd.concat([gaps_df, df])

    return gaps_df


def score_gap(gap: pd.DataFrame, gap_len_defs: dict, spf_ranges: dict) -> float:
    """For an individual gap give it a score based on its duration and change in value

    Args:
        gap (pd.DataFrame): The gap information (row from gaps DataFrame)
        gap_len_defs (dict): Defined thresholds for short, medium and long gaps in data
        spf_ranges (dict): Acceptable SPF max and min values for different durations

    Returns:
        float: The score for the given gap
    """
    duration = gap["Gap duration"]
    sensor = gap["sensor_type"]
    value_delta = gap[sensor]

    # If the meter has decreased we score all sensors the same
    # We expect this is due to a meter breaking and being replaced
    if value_delta < 0:
        if duration >= gap_len_defs["long"]:
            return 5.0
        elif duration >= gap_len_defs["medium"]:
            return 3.7
        elif duration >= gap_len_defs["short"]:
            return 3.1

    # Certain meters we expect may have long durations of flat data, or increase
    meter_list_A = [
        "Boiler_Energy_Output",
        "Immersion_Heater_Energy_Consumed",
        "Back-up_Heater_Energy_Consumed",
        "Circulation_Pump_Energy_Consumed",
    ]
    if sensor in meter_list_A:
        if value_delta >= 0:
            if duration >= gap_len_defs["long"]:
                return 3.0
            elif duration >= gap_len_defs["medium"]:
                return 2.0
            elif duration >= gap_len_defs["short"]:
                return 1.0

    # Gaps in heat pump consumed and output, we want to check these are sensible given what the other sensors are doing
    meter_list_B = ["Whole_System_Energy_Consumed", "Heat_Pump_Energy_Output"]
    if sensor in meter_list_B:
        # If either of the Heat Pump meters is missing for this gap, we expect there is another gap for the same time (+a little more) for that meter for which the two HP meter readings are present
        # In this case, we therefore don't score the gap too harshly. The other gap, for the other sensor, for which there is more data, will also score this time period (and will be harsh if appropriate)
        if gap[["Whole_System_Energy_Consumed", "Heat_Pump_Energy_Output"]].isna().any():
            if duration >= gap_len_defs["long"]:
                return 3.0
            elif duration >= gap_len_defs["medium"]:
                return 2.0
            elif duration >= gap_len_defs["short"]:
                return 1.0
        # If the meter has increased across the gap we want to calculate the SPFH2 is within expected range, if not: score more harshly
        if value_delta > 0:
            df = qa.calc_spf_for_all_windows(pd.DataFrame(gap).T)
            spf = df["spfh2"]
            spf_max = spf_ranges["medium"]["max"]
            spf_min = spf_ranges["medium"]["min"]
            if (spf.values[0] <= spf_max) & (spf.values[0] >= spf_min):
                if duration >= gap_len_defs["long"]:
                    return 3.0
                elif duration >= gap_len_defs["medium"]:
                    return 2.0
                elif duration >= gap_len_defs["short"]:
                    return 1.0
            else:
                if duration >= gap_len_defs["long"]:
                    return 5.0
                elif duration >= gap_len_defs["medium"]:
                    return 3.6
                elif duration >= gap_len_defs["short"]:
                    return 2.0
        # If the meter did not increase across the gap we want to check if other meters did and score accordingly
        elif value_delta == 0:
            # We must differentiate between the two heat pump sensors as they are not completely equivalent
            # Whole_System_Energy_Consumed is the meter for the whole system (including energy to circ pump, immersion heater, back-up heater etc.)
            # Heat_Pump_Energy_Output is the heat output from the heat pump
            meter_list_A = list(set(meter_list_A).intersection(set(gap.index)))
            other_HP_meter = list(set(meter_list_B) - set([sensor]))
            if sensor == "Whole_System_Energy_Consumed":
                # System consumed is flat, so if all other meters are flat that is what we expect
                all_flat = (gap[meter_list_A + other_HP_meter] == 0).all()

                if all_flat:
                    if duration >= gap_len_defs["long"]:
                        return 4.0
                    elif duration >= gap_len_defs["medium"]:
                        return 2.0
                    elif duration >= gap_len_defs["short"]:
                        return 1.0
                else:
                    if duration >= gap_len_defs["long"]:
                        return 5.0
                    elif duration >= gap_len_defs["medium"]:
                        return 3.5
                    elif duration >= gap_len_defs["short"]:
                        return 2.0

            # sensor == "Heat_Pump_Energy_Output" (This is the only other option)
            else:
                # If output is flat, then we need energy consumed by HP ONLY to be flat too
                consumed_HP_only = gap[other_HP_meter] - gap[meter_list_A].sum()
                consumed_HP_only_flat = (consumed_HP_only == 0).values[0]

                # If some increase apart from other HP reading
                if consumed_HP_only_flat:
                    if duration >= gap_len_defs["long"]:
                        return 3.0
                    elif duration >= gap_len_defs["medium"]:
                        return 2.0
                    elif duration >= gap_len_defs["short"]:
                        return 1.0
                else:
                    if duration >= gap_len_defs["long"]:
                        return 5.0
                    elif duration >= gap_len_defs["medium"]:
                        return 3.4
                    elif duration >= gap_len_defs["short"]:
                        return 2.0


def score_all_gaps(
    data: pd.DataFrame, gap_len_defs: dict, spf_ranges: dict, alteration_record: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify and score all the gaps in the given data

    Args:
        data (pd.DataFrame): The data to assess.
        gap_len_defs (dict): Defined thresholds for short, medium and long gaps in data
        spf_ranges (dict): Acceptable SPF max and min values for different durations
        alteration_record (pd.DataFrame): Record of alterations made to the data

    Returns:
        pd.DataFrame: The data returned with
    """

    gaps_df = find_gaps(data, gap_len_defs["short"])
    # Use gap duration and delta to score each row in each gap
    full_data = fill_gaps_with_nans(data)

    # Offset needed to ensure gaps are indexed correctly
    time_offset = full_data.index[1] - full_data.index[0]

    # Create a wide dataframe to store the gap scores in, as that is significantly
    # faster - particularly if there are lots of gaps.
    # We'll then melt it and join it back onto full_data.
    gap_scores = full_data.pivot(columns="sensor_type", values="value")
    for col in gap_scores.columns:
        gap_scores[col].values[:] = 0.0

    gap_inds = gaps_df.index
    for ind in gap_inds:
        gap = gaps_df.loc[ind]
        end_time = gap["Timestamp"] - time_offset
        start_time = gap["Timestamp"] - gap["Gap duration"] + time_offset
        gap_score = score_gap(gap, gap_len_defs, spf_ranges)
        gap_scores.loc[start_time:end_time, gap["sensor_type"]] = gap_score

    gap_scores = gap_scores.melt(value_name="gap_score", ignore_index=False)
    gap_scores = gap_scores.reset_index().set_index(["sensor_type", "Timestamp"])
    full_data = full_data.reset_index().set_index(["sensor_type", "Timestamp"])

    # This is slightly faster than full_data = full_data.join(gap_scores), but
    # relies on the rows being in the same order, which I think they should be
    # given the way we pivot and index above.
    full_data["gap_score"] = gap_scores["gap_score"]

    full_data = full_data.reset_index().set_index("Timestamp")
    return full_data, gaps_df


def score_all_data(data: pd.DataFrame, spf_ranges: dict, freq: str = "M", duration: str = "medium") -> pd.DataFrame:
    """Score the data based on performance factor being with acceptable range

    Args:
        data (pd.DataFrame): The data to score.
        spf_ranges (dict): Acceptable SPF max and min values for different durations
        freq (str): Frequency at which to score the data. Defaults to "M".
        duration (str): Duration to use for the spf_ranges. Defaults to "medium"

    Returns:
        pd.DataFrame: The scored data
    """
    # Need a Timestamp for the period the data is in (i.e. which month/day is this data point from [depending on freq])
    if "Timestamp" in data.columns:
        data["Period"] = data["Timestamp"].dt.to_period(freq)
    else:
        data["Period"] = data.index.to_period(freq)

    # Find the metered usage for each month
    resampled_grp = data.groupby(["sensor_type", "Period"])
    resampled_max = resampled_grp.max().reset_index()
    resampled_min = resampled_grp.min().reset_index()

    # This method for diff may miss some data used in the two minutes between one month and the next, so we prefer to use the diff of the max
    # This will only be used to fill any values which are NaN using diff() of the max (i.e. the first values for each sensor)
    resampled_diff = resampled_max.copy()
    resampled_diff["value"] = resampled_max["value"] - resampled_min["value"]
    resampled_diff = resampled_diff.pivot(index="Period", columns="sensor_type", values="value")

    resampled_max = resampled_max.pivot(index="Period", columns="sensor_type", values="value")
    usage_data = resampled_max.diff()

    usage_data = usage_data.fillna(resampled_diff)

    # Calculate the spf h2 for each period
    # spf calc requires all columns are present, if they are not we want to add them with zeros
    expected_cols = qa.spfh_required_cols["spfh2"]
    cols_to_add = list(set(expected_cols) - set(usage_data.columns))
    for col in cols_to_add:
        usage_data[col] = 0
    usage_data["spfh2"] = qa.spf(usage_data, metric="spfh2")

    spf_max = spf_ranges[duration]["max"]
    spf_min = spf_ranges[duration]["min"]

    # Find where spf is out of expected range
    spf_mask = data["Period"].isin(
        usage_data.loc[(usage_data["spfh2"] > spf_max) | (usage_data["spfh2"] < spf_min)].index
    )

    # Find where all data is flat for the period
    flat_mask = data["Period"].isin(usage_data.loc[(usage_data.drop(columns=["spfh2"]) < 1).all(axis=1)].index)

    # Score the data
    data["data_score"] = 0
    data.loc[flat_mask, "data_score"] = 3.2
    data.loc[spf_mask, "data_score"] = 3.3

    data = data.drop(columns=["Period"])

    return data


def level_resets(data: pd.DataFrame, alteration_record: pd.DataFrame) -> pd.DataFrame:
    """Where value decreases over gaps in data we assume meter has been replaced but we trust the data after the gap.
        We want to match level of the data across the gap so that it is still cumulative over the whole range.

    Args:
        data (pd.DataFrame): The data to reset the level
        alteration_record (pd.DataFrame): Record of alterations made to the data

    Returns:
        pd.DataFrame: The data after the levels has been reset
    """
    # Average power usage of largest element in any system (boiler) 35kW * 2 for realignment of timestamps (two points 00:00:01 and 00:03:59 would round to 00:02:00 and 00:04:00 [2 mins apart] but were more like 4 mins apart)
    avg_power_limit = 35 * 2
    # Can hit an error from duplicate indexing if we don't reset the index
    data = data.reset_index()
    # Calculate average power and previous values
    sensor_group = data.dropna(subset="value").groupby("sensor_type")
    data["previous_value"] = sensor_group["value"].shift(1)
    data["dv"] = sensor_group["value"].diff()
    data["dt"] = sensor_group["Timestamp"].diff() / pd.Timedelta(hours=1)
    data["avg_power_kW"] = data["dv"] / data["dt"]

    # Flag as a meter fault if: dv < 0 AND previous value > 0 AND -dv > 0.95*previous value
    # This should handel:
    # Meter resets
    data["meter_fault"] = (
        (data["dv"] < 0) & (data["previous_value"] > 0) & (-data["dv"] > 0.95 * data["previous_value"])
    )

    # Flag as meter fault if: |average power| > average power limit
    # This should handel:
    # Meter upward jumps
    # Anomalies (causing a jump both up then down)
    data["meter_fault"] = (abs(data["avg_power_kW"]) > avg_power_limit) | data["meter_fault"]

    # Calculate the required additions to account for the meter faults
    # We keep all diffs flagged as meter faults
    data.loc[data["meter_fault"], "fault_diff"] = -data.loc[data["meter_fault"], "dv"]
    data.loc[~data["meter_fault"], "fault_diff"] = 0
    # Groupby sensor_type then cumsum will give the required alteration for all points
    data["alteration"] = data.groupby("sensor_type")["fault_diff"].cumsum()

    # Correct the data so the meter reset is flat
    data["value"] = data["value"] + data["alteration"]

    # Add to the alteration record for each meter reset correction preformed
    # for index, row in data.loc[~(data["fault_diff"] == 0)].iterrows():
    #     changed_data = data.loc[data["sensor_type"] == row["sensor_type"]]
    #     changed_data = changed_data.loc[index:]
    #     alteration_record = add_alteration_record(
    #         alteration_record,
    #         changed_data=changed_data,
    #         alteration=row["fault_diff"],
    #         reason="Meter fault correction - alteration is absolute change applied",
    #     )

    faults = data.loc[~(data["fault_diff"] == 0)]

    starts = faults["Timestamp"]
    sensor_types = faults["sensor_type"]
    ends_dict = {
        sensor: data.loc[data["sensor_type"] == sensor, "Timestamp"].max() for sensor in sensor_types.unique()
    }
    ends = [ends_dict[sensor] for sensor in sensor_types.values]
    alterations = faults["fault_diff"]

    new_alteration = {
        "sensor_type": sensor_types,
        "start_time": starts,
        "end_time": ends,
        "alteration": alterations,
    }
    new_alterations_df = pd.DataFrame(new_alteration)
    new_alterations_df["reason"] = "Meter fault correction - alteration is absolute change applied"

    alteration_record = pd.concat([alteration_record, new_alterations_df])

    # Calculate the drops that we are not correcting for - This is temporary for testing purposes
    data["drops_not_reset"] = (abs(data["avg_power_kW"]) < avg_power_limit) & (data["dv"] < 0)
    data["drops_not_reset_diff"] = -data.loc[data["drops_not_reset"], "dv"]
    drops_not_reset = data.loc[data["drops_not_reset"]]
    # Add to the alteration record for all drops that were not corrected
    if len(drops_not_reset) > 0:
        alteration_record = add_alteration_record(
            alteration_record,
            changed_data=drops_not_reset,
            alteration=f"""max:{drops_not_reset["dv"].max()} - min:{drops_not_reset["dv"].min()} - count:{drops_not_reset["dv"].count()}""",
            reason="Drops not reset stats",
        )

    data = data.set_index("Timestamp")
    data = data.drop(
        columns=[
            "dv",
            "dt",
            "avg_power_kW",
            "previous_value",
            "meter_fault",
            "fault_diff",
            "alteration",
            "drops_not_reset",
            "drops_not_reset_diff",
        ]
    )

    return data, alteration_record


def correct_reversed_meter(data: pd.DataFrame, alteration_record: pd.DataFrame):
    """Some meters have been installed the wrong way round, this means cumulative readings decrease over time.
    To fix this we will check if the reading decreases on a majority of days, if it does, we will reverse the
    meter readings.

    Args:
        data (pd.DataFrame): The cumulative meter data
        alteration_record (pd.DataFrame): Record of alterations made to the data

    Returns:
        _type_: The corrected cumulative meter data
    """

    sensor_list = data["sensor_type"].unique()
    for sensor in sensor_list:
        sensor_mask = data["sensor_type"] == sensor
        # Calculate the daily difference in the meter reading
        diff = data.loc[sensor_mask].copy()
        diff["value"] = diff["value"].diff().dropna()
        daily_diff = resample_data(diff, freq="d", agg_fn="sum")
        # If daily changes are mostly negative then we want to reverse the meter reading
        decrease_count = (daily_diff["value"] < 0).sum()
        increase_count = (daily_diff["value"] > 0).sum()
        if decrease_count > increase_count:
            data.loc[sensor_mask, "value"] = -data.loc[data["sensor_type"] == sensor, "value"]
            alteration_record = add_alteration_record(
                alteration_record,
                changed_data=data.loc[sensor_mask],
                alteration="Value * -1",
                reason="Cumulative meter decreasing",
            )

    return data, alteration_record


def find_windows(data: pd.DataFrame, window_len_mths: int = 12) -> pd.DataFrame:
    """find and score windows in the data

    Args:
        data (pd.DataFrame): The data to find windows in, this data should have its gaps scored
        window_len_mths (pd.Timedelta, optional): The duration of the windows. Defaults to pd.Timedelta(days=365).

    Returns:
        pd.DataFrame: DataFrame containing the details of the available windows in the data.
    """
    # If the data doesn't contain the two heat pump energy values then we can't calculate SPFs, so we shouldn't find any windows
    if ("Whole_System_Energy_Consumed" not in data["sensor_type"].unique()) | (
        "Heat_Pump_Energy_Output" not in data["sensor_type"].unique()
    ):
        return pd.DataFrame()

    # We want to assess the data over a period but we can't use windows that start in the middle of a gap
    # Lets assume we will use times where the required data is present at the start of the day
    # So here we keep only the rows where all the data is present
    df = (
        data.dropna(subset="value")[["sensor_type", "value"]]
        .pivot(columns="sensor_type")
        .droplevel(0, axis=1)
        .dropna()
    )
    duration = pd.DateOffset(months=window_len_mths)

    timestamps = pd.to_datetime(pd.Series(df.index))

    # We want to start and end at the start of each day (midnight, 00:00)
    midnight_mask = (timestamps.dt.minute == 0) & (timestamps.dt.hour == 0)

    # Start times must be more than the duration from the last timestamp
    start_mask = midnight_mask & (timestamps <= (timestamps.max() - duration))
    start_times = timestamps.loc[start_mask]

    # End times must be more than the duration from the first timestamp
    end_mask = midnight_mask & (timestamps >= (timestamps.min() + duration))
    end_times = timestamps.loc[end_mask]

    # We want windows where data is present at start time and corresponding end time
    start_times = pd.Series(
        [start for start in start_times if start + duration in end_times.values], dtype="datetime64[ns]",
    )

    # If there are any valid time windows then we calculate the score etc.
    if len(start_times) > 0:
        windows = pd.DataFrame(start_times, columns=["start"])
        windows["end"] = start_times + duration
        score_list = ["gap_score", "data_score", "score"]
        data = data.reset_index()
        score_data_group = data.groupby("Timestamp")[score_list]
        score_data_group_max = score_data_group.max()
        # The sum and the count are required here rather than the mean for two reasons:
        # 1.  there may be cases (usually at the start) where some sensors are not present and
        #     so NaN values exist, and so the mean of the mean for each timestamp is not the
        #     same as the mean of all valid data points over the window.
        # 2.  the mean of the mean leads to floating point differences as well
        score_data_group_sum = score_data_group.sum()
        score_data_group_count = score_data_group.count()

        windows[[f"max_{score}" for score in score_list]] = windows.apply(
            lambda x: score_data_group_max.loc[x["start"] : x["end"]].max(), axis=1,
        )
        windows[[f"mean_{score}" for score in score_list]] = windows.apply(
            lambda x: score_data_group_sum.loc[x["start"] : x["end"]].sum()
            / score_data_group_count.loc[x["start"] : x["end"]].sum(),
            axis=1,
        )
        windows["end"] = windows["end"]
        sensor_list = df.columns
        windows[[f"{sensor}" for sensor in sensor_list]] = windows.apply(
            lambda x: df.loc[x["end"]] - df.loc[x["start"]], axis=1,
        )

        windows["mean_score"] = round(windows["mean_score"], 10)

    else:
        windows = pd.DataFrame()

    windows = qa.calc_spf_for_all_windows(windows)

    return windows


def find_best_window(
    data: pd.DataFrame,
    gap_len_defs: dict,
    home_summary_part: pd.DataFrame,
    spf_ranges: dict,
    window_len_mths: int = 12,
    plot_full_save_path: str = "",
    plot_window_save_path: str = "",
    location_out_cleaned: str = "",
    file: str = "",
    save_scored_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find the window in the data with the best quality data.

    Args:
        data (pd.DataFrame): Data in which to find the window
        gap_len_defs (dict): Dictionary to define the range of gap durations.
        home_summary_part (pd.DataFrame): DataFrame summarising outputs for this home, will be added to and output
        spf_ranges (dict): Acceptable SPF max and min values for different durations
        window_len_mths (int, optional): The length of the window in months. (Defaults to 12)
        plot_full_save_path (str, optional): Path at which to save the plot of the full data. Use "none" to show figure. Defaults to "".
        plot_window_save_path (str, optional): Path at which to save the plot of the window data. Use "none" to show figure. Defaults to "".

    Returns:
        pd.Timestamp: Timestamp for the end of the best window
        dict: Scores for the chosen window
        pd.DataFrame: home_summary_part updated with info on the cleaned data
        pd.DataFrame: Record of alterations made to the data
    """

    # Some functions following will change some of the data, we will make a record of these changes
    alteration_record = pd.DataFrame()

    # These 6 functions have been commented out as they are now in the cleaning algorithm and not in window selection. To be deleted if everything works.
    # data, alteration_record = correct_reversed_meter(data, alteration_record)

    # data, alteration_record = remove_anomalous_points(data, alteration_record)

    # Score the data based on if monthly spfh2 is within acceptable range or if all data is flat

    temp_sensors = [sensor for sensor in data["sensor_type"].unique() if sensor.endswith("Temperature")]
    temp_data = data.loc[data["sensor_type"].isin(temp_sensors)]

    data = qa.select_sensors(data, metric="spfh2")

    data = score_all_data(data, spf_ranges)

    data, gaps_df = score_all_gaps(data, gap_len_defs, spf_ranges, alteration_record)

    if ("gap_score" in data.columns) and ("data_score" in data.columns):
        data["score"] = data[["gap_score", "data_score"]].max(axis=1)

    # # Some periods have output heat much lower than reasonable, but this is fixed after some time.
    # # We expect this is caused by the heat meter not working, we want to remove this data
    # data, alteration_record = remove_start_with_spf_out_of_range(data, spf_ranges, alteration_record)

    # # If a meter breaks and is replaced we see a gap in the data where the value drops
    # # We handle this here
    # data, alteration_record = level_resets(data, alteration_record)

    # Find windows where there is data at both the start and the end
    windows = find_windows(data, window_len_mths)

    data = pd.concat([data, temp_data])

    if save_scored_data:
        data.reset_index().to_parquet(os.path.join(location_out_cleaned, "cleaned_scored", file), index=False)

    score_list = ["gap_score", "data_score", "score"]
    scores = dict()
    # Only try to find the best window if we have found any
    if len(windows) > 0:
        # Find the windows with the lowest (minimum) max score
        best_windows = windows.loc[windows["max_score"] == windows["max_score"].min()]
        # From these, find the windows with the lowest (minimum) mean gap score
        best_windows = best_windows.loc[best_windows["mean_score"] == best_windows["mean_score"].min()]
        # From these, take the most recent one (in case there are several with the same score)
        final_window = best_windows.loc[best_windows["end"] == best_windows["end"].max()]
        end_timestamp = pd.Timestamp(final_window["end"].values[0])

        import warnings

        warnings.filterwarnings("ignore")

        if len(best_windows) > 0:
            # We want to add statistics for all windows with max score < 4
            acceptable_windows = windows.loc[windows["max_score"] < 4]
            low_score_window_stats = acceptable_windows[["spfh2", "spfh3", "spfh4"]].describe()
            col_list = low_score_window_stats.columns
            stat_list = low_score_window_stats.index
            for col in col_list:
                for stat in stat_list:
                    home_summary_part[f"acceptable windows: {col}: {stat}"] = low_score_window_stats.loc[stat, col]

        import warnings

        warnings.filterwarnings("default")

        # Plot the full data range
        if plot_full_save_path != "":
            plot_data(
                data, window_len_mths=window_len_mths, end_timestamp=end_timestamp, path=plot_full_save_path,
            )

        # Plot the window data only
        if plot_window_save_path != "":
            # Keep the data only in the window
            window_mask = (data.index >= (end_timestamp - pd.DateOffset(months=window_len_mths))) & (
                data.index <= end_timestamp
            )
            window_data = data.loc[window_mask]

            # Reset the data values to start at 0 from the start of the window
            sensor_types = window_data["sensor_type"].unique()
            for sensor in sensor_types:
                sensor_mask = window_data["sensor_type"] == sensor
                window_data.loc[sensor_mask, "value"] = (
                    window_data.loc[sensor_mask, "value"] - window_data.loc[sensor_mask, "value"].iloc[0]
                )

            # Generate the plot
            plot_data(
                window_data, window_len_mths=None, end_timestamp=None, path=plot_window_save_path,
            )
    else:
        final_window = pd.DataFrame()
        windows = pd.DataFrame()

        # If no window is found, we would still like a plot of the data, just without the rectangle for the window
        if plot_full_save_path != "":
            plot_data(data, window_len_mths=None, end_timestamp=None, path=plot_full_save_path)

    return (
        final_window,
        home_summary_part,
        windows,
        alteration_record,
        data[["sensor_type", "value"]],
    )


def plot_data(
    data: pd.DataFrame, window_len_mths: int = None, end_timestamp: pd.Timestamp = None, path: str = "",
) -> go.Figure:
    """Plots the full range of the data with a rectangle overlay showing the chosen window. A subplot below the
    data plot displays the gap score as a color plot.

    Args:
        data (pd.DataFrame): The data to be plotted
        window_len_mths (int): The duration of the selected window
        end_timestamp (pd.Timestamp): The end timestamp of the selected window
        path (str): The file path at which to save the plot

    Returns:
        go.Figure: figure object for the plot
    """
    # Plot data, with window and gap_score
    plot_data = resample_data(data, freq="1D")
    plot_data = plot_data.reset_index()
    fig = px.scatter(plot_data, x="Timestamp", y="value", color="sensor_type")

    trace1 = list(fig.select_traces())

    plot_score = plot_data[["Timestamp", "gap_score", "data_score", "score"]].groupby("Timestamp").mean()
    fig = px.imshow(plot_score.T)

    trace2 = list(fig.select_traces())

    fig = make_subplots(rows=2, shared_xaxes=True, row_width=[0.2, 0.8], vertical_spacing=0.02)
    fig.add_traces(trace1, rows=1, cols=1)
    fig.add_traces(trace2, rows=2, cols=1)
    if (window_len_mths is not None) & (end_timestamp is not None):
        # Only plot the rectangle to show the window if we have the info for it
        fig.add_vrect(x0=end_timestamp - pd.DateOffset(months=window_len_mths), x1=end_timestamp)
    fig["layout"].update(
        height=400,
        width=1000,
        margin=dict(l=50, r=20, t=40, b=10),
        coloraxis={"showscale": False, "cmax": 5, "cmin": 0, "colorscale": "matter"},
    )

    if path != "none":
        pio.write_image(fig, path, format="png", engine="orca")
    else:
        fig.show()

    return fig
