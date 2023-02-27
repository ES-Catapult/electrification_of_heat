import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import analysis_functions as af
import numpy as np


def colour_scheme():
    """Contains the colour scheme for ESC in the EoH project

    Returns:
        _type_: Dictionary of colours as RGB
    """
    colours = {
        "colour1_rgb": (0, 170, 149),
        "colour2_rgb": (236, 124, 48),
        "colour3_rgb": (233, 192, 40),
        "colour4_rgb": (148, 187, 75),
        "colour5_rgb": (78, 151, 224),
    }

    return colours


colour1_rgb = colour_scheme().get("colour1_rgb")
colour2_rgb = colour_scheme().get("colour2_rgb")
colour3_rgb = colour_scheme().get("colour3_rgb")
colour4_rgb = colour_scheme().get("colour4_rgb")
colour5_rgb = colour_scheme().get("colour4_rgb")


def chart_homes_waterfall(
    data: pd.DataFrame,
    SPF_value_min_threshold: float,
    SPF_value_max_threshold: float,
    window_max_score_threshold: int,
    still_monitoring_cut_off: pd.Timestamp,
):
    """Creates waterfall chart showing the number of homes at each stage of the filtering process.
    Adds bool columns to the dataframe to show which homes were removed from the analysis and at which point.

    Args:
        data (pd.DataFrame): Dataframe to be used to create waterfall and be supplemented with bool columns.
        SPF_value_min_threshold (float): Minimum acceptable value for all SPF types
        SPF_value_max_threshold (float): Maximum acceptable value for all SPF types
        window_max_score_threshold (int): The data quality score threshold that the data must not exceed
        still_monitoring_cut_off (pd.Timestamp): If the end date over the all data for a home is before this date, it could be considered as no longer monitoring.
    """

    # Make sure Whole_end is in correct datetime format
    data["Whole_end"] = pd.to_datetime(data["Whole_end"])

    # Make a copy of the dataframe to filter for the waterfall chart
    homes_output = data.copy()

    # Convert dataframe to long format
    homes_output = af.reformat_SPF_to_long(homes_output, cold_analysis=False)

    # Set the 6 different exclusions that will be applied to the dataframe to calculate the numbers for the waterfall

    # At least 365 days of data and beyond the cut-off
    exclusion_1 = (homes_output["Whole_duration_days"] >= 365) | (
        homes_output["Whole_end"] >= still_monitoring_cut_off
    )

    # Tail end of data is before the cut-off
    exclusion_2 = (homes_output["Whole_duration_days"] >= 365) | (homes_output["Whole_end"] < still_monitoring_cut_off)

    # Cleaned data has more than 365 days of data
    exclusion_3 = homes_output["Cleaned_duration_days"] >= 365

    # Has a valid spf value calculated
    exclusion_4 = homes_output["SPF_value"].isna() == False

    # Is within the analysis thresholds
    exclusion_5 = (homes_output["SPF_value"] >= SPF_value_min_threshold) & (
        homes_output["SPF_value"] <= SPF_value_max_threshold
    )

    # Data quality score is within an acceptable range
    exclusion_6 = homes_output["window_max_score"] < window_max_score_threshold

    # Create exclusion columns
    homes_output["<1yr of data overall (monitoring stopped)"] = ~exclusion_1
    homes_output["<1yr of data (ongoing monitoring)"] = ~exclusion_2
    homes_output["<1yr of usable data (early monitoring issues)"] = ~exclusion_3
    homes_output["No valid 1yr window due to gaps"] = ~exclusion_4
    homes_output[f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}"] = ~exclusion_5

    # Reduce to 1 row per home and remove the home if even 1 of its spf types is out of the threshold
    SPF_homes_removed = (
        homes_output.groupby("Property_ID")[
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}"
        ].max()
        != 0
    )
    homes_output = homes_output.merge(SPF_homes_removed, on="Property_ID", suffixes=("_x", "")).drop(
        f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}_x", axis=1
    )

    # Continuation of creating exclusion columns
    homes_output["Quality score outside threshold"] = ~exclusion_6

    # Sum all exclusions
    homes_output["Removed_from_analysis"] = homes_output[
        [
            "<1yr of data overall (monitoring stopped)",
            "<1yr of data (ongoing monitoring)",
            "<1yr of usable data (early monitoring issues)",
            "No valid 1yr window due to gaps",
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}",
            "Quality score outside threshold",
        ]
    ].sum(axis=1)

    # If sum of exclusions equals zero then it is included in the analysis (Removed_from_analysis=False)
    # otherwise not (Removed_from_analysis=True)
    homes_output["Removed_from_analysis"] = np.where(homes_output["Removed_from_analysis"] == 0, False, True)

    # Revert back to 1 row per home ready for merge
    homes_exclusion = homes_output.groupby("Property_ID")[
        [
            "<1yr of data overall (monitoring stopped)",
            "<1yr of data (ongoing monitoring)",
            "<1yr of usable data (early monitoring issues)",
            "No valid 1yr window due to gaps",
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}",
            "Quality score outside threshold",
            "Removed_from_analysis",
        ]
    ].first()

    # Join exclusion fields onto original dataframe
    data = data.merge(homes_exclusion, on="Property_ID")

    # Waterfall

    # 2 E.On were never connected due to internet problems, these were Property_IDs: 321164 & 326442
    # We therefore add 2 to the 'total_installed' value
    # We also add 2 more at the moment as an interim adjustment so the total is as expected

    # 1st bar on waterfall
    total_installed = homes_output["Property_ID"].unique().size + 4

    # 2nd bar on waterfall
    homes_output = homes_output.loc[exclusion_1]
    lt_1yr_stopped_monitoring = homes_output["Property_ID"].unique().size

    # 3rd bar on waterfall
    homes_output = homes_output.loc[exclusion_2]
    lt_1yr_monitoring = homes_output["Property_ID"].unique().size

    # 4th bar on waterfall
    homes_output = homes_output.loc[exclusion_3]
    lt_1yr_usable = homes_output["Property_ID"].unique().size

    # 5th bar on waterfall
    homes_output = homes_output.loc[exclusion_4]
    no_valid_1yr_window = homes_output["Property_ID"].unique().size

    # 6th bar on waterfall
    homes_output = homes_output.loc[exclusion_5]
    # Remove the entire home if at least one SPF_type was lost in threshold filtering
    homes_output = af.remove_homes_without_all_SPF_types(homes_output)
    SPF_not_in_range = homes_output["Property_ID"].unique().size

    # 7th bar on waterfall
    homes_output = homes_output.loc[exclusion_6]
    window_score_not_in_range = homes_output["Property_ID"].unique().size

    # Create waterfall dataframe
    homes_waterfall = pd.DataFrame(
        [
            {
                "Total installed": total_installed,
                "<1yr of data overall (monitoring stopped)": lt_1yr_stopped_monitoring,
                "<1yr of data (ongoing monitoring)": lt_1yr_monitoring,
                "<1yr of usable data (early monitoring issues)": lt_1yr_usable,
                "No valid 1yr window due to gaps": no_valid_1yr_window,
                f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}": SPF_not_in_range,
                "Quality score outside threshold": window_score_not_in_range,
            }
        ]
    )

    # Take difference for the waterfall
    homes_waterfall = homes_waterfall.diff(axis=1)
    homes_waterfall = homes_waterfall.fillna({"Total installed": total_installed}).astype(int)

    # 8th bar on waterfall
    remaining_homes = int(homes_waterfall.sum(axis=1))
    homes_waterfall["Total remaining"] = homes_waterfall.sum(axis=1)

    # 9th bar on waterfall
    homes_waterfall["Total available"] = lt_1yr_stopped_monitoring

    # Reorder waterfall bars
    homes_waterfall = pd.concat(
        [
            homes_waterfall.iloc[:, 0:2],
            homes_waterfall["Total available"],
            homes_waterfall.iloc[:, 2:8],
        ],
        axis=1,
    )

    # Create waterfall chart
    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=[
                "relative",
                "relative",
                "total",
                "relative",
                "relative",
                "relative",
                "relative",
                "relative",
                "total",
            ],
            x=homes_waterfall.columns,
            textposition="auto",
            text=homes_waterfall.iloc[0, :],
            y=homes_waterfall.iloc[0, :],
            totals={"marker": {"color": f"rgb{colour3_rgb}"}},
            increasing={"marker": {"color": f"rgb{colour1_rgb}"}},
            decreasing={"marker": {"color": f"rgb{colour2_rgb}"}},
            connector={"line": {"color": f"rgb{colour4_rgb}"}},
        )
    )

    waterfall.update_layout(title="Homes Inclusion Waterfall", width=800, height=500)
    waterfall.update_xaxes(automargin=True)
    waterfall.update_xaxes(
        tickangle=30,
    )

    waterfall.show()

    # Print auxillary info
    print(f"There are {total_installed} total homes with heat pump installations, of which:")
    print(
        f"{total_installed - lt_1yr_stopped_monitoring} homes are lost as they do not have 12 months of data and are no longer receiving new data"
    )
    print(
        f"{lt_1yr_stopped_monitoring - lt_1yr_monitoring} further homes are lost as they do not have 12 months of data yet"
    )
    print(
        f"{lt_1yr_monitoring - lt_1yr_usable} further homes are lost as they do not have 12 months of data once the data is cleaned"
    )
    print(
        f"{lt_1yr_usable - no_valid_1yr_window} further homes are lost as there are no valid 12 month windows to calculate the SPF from"
    )
    print(
        f"{no_valid_1yr_window - SPF_not_in_range} further homes are lost as they fall out of the threshold: {SPF_value_min_threshold} <= SPF_value <= {SPF_value_max_threshold}"
    )
    print(
        f"{SPF_not_in_range - window_score_not_in_range} further homes are lost as they fall out of the threshold: window_max_score < {window_max_score_threshold}"
    )
    print(f"{remaining_homes} homes remain")

    # return data


def chart_homes_waterfall2(
    data: pd.DataFrame,
    SPF_value_min_threshold: float,
    SPF_value_max_threshold: float,
    window_max_score_threshold: int,
    still_monitoring_cut_off: pd.Timestamp,
):
    """Creates waterfall chart showing the number of homes at each stage of the filtering process.
    Adds bool columns to the dataframe to show which homes were removed from the analysis and at which point.

    Args:
        data (pd.DataFrame): Dataframe to be used to create waterfall and be supplemented with bool columns.
        SPF_value_min_threshold (float): Minimum acceptable value for all SPF types
        SPF_value_max_threshold (float): Maximum acceptable value for all SPF types
        window_max_score_threshold (int): The data quality score threshold that the data must not exceed
        still_monitoring_cut_off (pd.Timestamp): If the end date over the all data for a home is before this date, it could be considered as no longer monitoring.

    Returns:
        _type_: Homes_output with added exclusion columns
    """

    # Make sure Whole_end is in correct datetime format
    data["Whole_end"] = pd.to_datetime(data["Whole_end"])

    # Make a copy of the dataframe to filter for the waterfall chart
    homes_output = data.copy()

    # Convert dataframe to long format
    homes_output = af.reformat_SPF_to_long(homes_output, cold_analysis=False)

    # Set the 6 different exclusions that will be applied to the dataframe to calculate the numbers for the waterfall

    # At least 365 days of data and beyond the cut-off
    exclusion_1 = (homes_output["Whole_duration_days"] >= 365) | (
        homes_output["Whole_end"] >= still_monitoring_cut_off
    )

    # Tail end of data is before the cut-off
    exclusion_2 = (homes_output["Whole_duration_days"] >= 365) | (homes_output["Whole_end"] < still_monitoring_cut_off)

    # Cleaned data has more than 365 days of data
    exclusion_3 = homes_output["Cleaned_duration_days"] >= 365

    # Has a valid spf value calculated
    exclusion_4 = homes_output["SPF_value"].isna() == False

    # Is within the analysis thresholds
    exclusion_5 = (homes_output["SPF_value"] >= SPF_value_min_threshold) & (
        homes_output["SPF_value"] <= SPF_value_max_threshold
    )

    # Data quality score is within an acceptable range
    exclusion_6 = homes_output["window_max_score"] < window_max_score_threshold

    # Create exclusion columns
    homes_output["<1yr of data"] = ~exclusion_1 | ~exclusion_2
    homes_output["<1yr of usable data (early monitoring issues)"] = ~exclusion_3
    homes_output["No valid 1yr window due to gaps"] = ~exclusion_4
    homes_output[f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}"] = ~exclusion_5

    # Reduce to 1 row per home and remove the home if even 1 of its spf types is out of the threshold
    SPF_homes_removed = (
        homes_output.groupby("Property_ID")[
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}"
        ].max()
        != 0
    )
    homes_output = homes_output.merge(SPF_homes_removed, on="Property_ID", suffixes=("_x", "")).drop(
        f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}_x", axis=1
    )

    # Continuation of creating exclusion columns
    homes_output["Quality score outside threshold"] = ~exclusion_6

    # Sum all exclusions
    homes_output["Removed_from_analysis"] = homes_output[
        [
            # "<1yr of data overall (monitoring stopped)",
            "<1yr of data",
            "<1yr of usable data (early monitoring issues)",
            "No valid 1yr window due to gaps",
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}",
            "Quality score outside threshold",
        ]
    ].sum(axis=1)

    # If sum of exclusions equals zero then it is included in the analysis (Removed_from_analysis=False)
    # otherwise not (Removed_from_analysis=True)
    homes_output["Removed_from_analysis"] = np.where(homes_output["Removed_from_analysis"] == 0, False, True)

    # Revert back to 1 row per home ready for merge
    homes_exclusion = homes_output.groupby("Property_ID")[
        [
            # "<1yr of data overall (monitoring stopped)",
            "<1yr of data",
            "<1yr of usable data (early monitoring issues)",
            "No valid 1yr window due to gaps",
            f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}",
            "Quality score outside threshold",
            "Removed_from_analysis",
        ]
    ].first()

    # Join exclusion fields onto original dataframe
    data = data.merge(homes_exclusion, on="Property_ID")

    # Waterfall

    # 2 E.On were never connected due to internet problems, these were Property_IDs: 321164 & 326442
    # We therefore add 2 to the 'total_installed' value.
    # We also add 2 more at the moment as an interim adjustment so the total is as expected

    # 1st bar on waterfall
    total_installed = homes_output["Property_ID"].unique().size + 4

    # 2nd bar on waterfall
    homes_output = homes_output.loc[exclusion_1]
    homes_output = homes_output.loc[exclusion_2]
    lt_1yr_monitoring = homes_output["Property_ID"].unique().size

    # 3rd bar on waterfall
    homes_output = homes_output.loc[exclusion_3]
    lt_1yr_usable = homes_output["Property_ID"].unique().size

    # 4th bar on waterfall
    homes_output = homes_output.loc[exclusion_4]
    no_valid_1yr_window = homes_output["Property_ID"].unique().size

    # 5th bar on waterfall
    homes_output = homes_output.loc[exclusion_5]
    # Remove the entire home if at least one SPF_type was lost in threshold filtering
    homes_output = af.remove_homes_without_all_SPF_types(homes_output)
    SPF_not_in_range = homes_output["Property_ID"].unique().size

    # 6th bar on waterfall
    homes_output = homes_output.loc[exclusion_6]
    window_score_not_in_range = homes_output["Property_ID"].unique().size

    # Create waterfall dataframe
    homes_waterfall = pd.DataFrame(
        [
            {
                "Total installed": total_installed,
                # "<1yr of data overall (monitoring stopped)": lt_1yr_stopped_monitoring,
                "<1yr of data": lt_1yr_monitoring,
                "<1yr of usable data (early monitoring issues)": lt_1yr_usable,
                "No valid 1yr window due to gaps": no_valid_1yr_window,
                f"SPF < {SPF_value_min_threshold} or SPF > {SPF_value_max_threshold}": SPF_not_in_range,
                "Quality score outside threshold": window_score_not_in_range,
            }
        ]
    )

    # Take difference for the waterfall
    homes_waterfall = homes_waterfall.diff(axis=1)
    homes_waterfall = homes_waterfall.fillna({"Total installed": total_installed}).astype(int)

    # 7th bar on waterfall
    remaining_homes = int(homes_waterfall.sum(axis=1))
    homes_waterfall["Total remaining"] = homes_waterfall.sum(axis=1)

    # Reorder waterfall bars
    homes_waterfall = pd.concat(
        [
            homes_waterfall.iloc[:, 0:2],
            homes_waterfall.iloc[:, 2:8],
        ],
        axis=1,
    )

    # Create waterfall chart
    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=[
                "relative",
                # "relative",
                # "total",
                "relative",
                "relative",
                "relative",
                "relative",
                "relative",
                "total",
            ],
            x=homes_waterfall.columns,
            textposition="auto",
            text=homes_waterfall.iloc[0, :],
            y=homes_waterfall.iloc[0, :],
            totals={"marker": {"color": f"rgb{colour3_rgb}"}},
            increasing={"marker": {"color": f"rgb{colour1_rgb}"}},
            decreasing={"marker": {"color": f"rgb{colour2_rgb}"}},
            connector={"line": {"color": f"rgb{colour4_rgb}"}},
        )
    )

    waterfall.update_layout(title="Homes Inclusion Waterfall v2", width=800, height=500)
    waterfall.update_xaxes(automargin=True)
    waterfall.update_xaxes(
        tickangle=30,
    )

    waterfall.show()

    # Print auxillary info
    print(f"There are {total_installed} total homes with heat pump installations, of which:")
    print(f"{total_installed - lt_1yr_monitoring} homes are lost as they do not have 12 months of data yet")
    print(
        f"{lt_1yr_monitoring - lt_1yr_usable} further homes are lost as they do not have 12 months of data once the data is cleaned"
    )
    print(
        f"{lt_1yr_usable - no_valid_1yr_window} further homes are lost as there are no valid 12 month windows to calculate the SPF from"
    )
    print(
        f"{no_valid_1yr_window - SPF_not_in_range} further homes are lost as they fall out of the threshold: {SPF_value_min_threshold} <= SPF_value <= {SPF_value_max_threshold}"
    )
    print(
        f"{SPF_not_in_range - window_score_not_in_range} further homes are lost as they fall out of the threshold: window_max_score < {window_max_score_threshold}"
    )
    print(f"{remaining_homes} homes remain")

    return data


def SPF_h2_threshold_scatter(
    data: pd.DataFrame,
    SPF_value_min_threshold: float,
    SPF_value_max_threshold: float,
    scatter_opacity: float,
):
    """Creates a scatter chart showing 1 point for each home while denoting what type of heat pump it is and whether it is included in the analysis or not

    Args:
        data (pd.DataFrame): homes_output dataframe containing 1 row per home
        SPF_value_min_threshold (float): Minimum acceptable value for all SPF types
        SPF_value_max_threshold (float): Maximum acceptable value for all SPF types
        scatter_opacity (float): Set opacity of lines and markers on scatter plots to allow high density scatters to display better
    """

    # Rename bool values to 'included' and 'removed' for clarity
    data["Removed_from_analysis"] = (
        data["Removed_from_analysis"].astype(str).replace({"False": "Included", "True": "Removed"})
    )

    # Rename field for clarity
    data = data.rename({"Removed_from_analysis": "Inclusion"}, axis=1)

    # Filter data for SPFH2 only
    fig_data = data[(data["SPF_type"] == "SPFH2")]

    # Drop homes where a window max score could not be calculated due to data quality
    fig_data = fig_data.dropna(subset="window_max_score")

    fig_scatter = px.scatter(
        fig_data,
        title="Heat Pump Energy Output vs SPFH2",
        x="window_Total_Energy_Output",
        y="SPF_value",
        color="HP_Type_2",
        height=620,
        width=800,
        symbol="Inclusion",
        symbol_sequence=["circle", "x-open"],
        range_y=(0, 8.5),
        range_x=(0, 32000),
        hover_name="Property_ID",
        labels={
            "window_Total_Energy_Output": "Total Energy Output (kWh)",
            "SPF_value": "SPFH2",
        },
    )

    # Add horizontal lines for spf min and max threshold for clarity
    fig_scatter.update_traces(opacity=scatter_opacity)

    fig_scatter.add_hline(
        y=SPF_value_min_threshold,
        line_dash="dash",
        line_width=1,
        annotation_text="SPF_min_threshold",
        annotation_position="bottom right",
    )
    fig_scatter.add_hline(
        y=SPF_value_max_threshold,
        line_dash="dash",
        line_width=1,
        annotation_text="SPF_max_threshold",
    )

    fig_scatter.show()
