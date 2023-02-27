import os
import qa_functions as qa
import pandas as pd
import utils
import data_scoring_fns as ds
import temperature_data_fns as td
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import numpy as np

sensors_may_swap = [
    "Hot_Water_Flow_Temperature",
    "Heat_Pump_Heating_Flow_Temperature",
    "Heat_Pump_Return_Temperature",
]

normal_ranges = {
    "Internal_Air_Temperature": {"max": 40, "min": 0},
    "External_Air_Temperature": {"max": 40.3, "min": -27.2},
    "Heat_Pump_Heating_Flow_Temperature": {"max": 80, "min": 5},
    "Heat_Pump_Return_Temperature": {"max": 80, "min": 5},
    "Hot_Water_Flow_Temperature": {"max": 80, "min": 5},
    "Brine_Flow_Temperature": {"max": 30, "min": -10},
    "Brine_Return_Temperature": {"max": 30, "min": -10},
}

non_temp_sensors = [
    "Boiler_Energy_Output",
    "Whole_System_Energy_Consumed",
    "Immersion_Heater_Energy_Consumed",
    "Back-up_Heater_Energy_Consumed",
    "Circulation_Pump_Energy_Consumed",
    "Heat_Pump_Energy_Output",
]


def classify_and_predict(location_out: str, files_list: str):
    # Using the temperature_stats.csv file, we create classifiers and make predictions
    # and create the homes_dropped subset of homes

    rng = np.random.RandomState(0)

    # If you don't run the cell above, you can load in the data it generates here
    path = os.path.join(location_out, "temperature_stats.csv")
    all_stats = pd.read_csv(path)

    # Drop null rows
    all_stats = all_stats[~all_stats.isnull().any(axis=1)]

    # Only interested in separating Hot water from HP flow and return temperatures for now
    # all_stats_backup = all_stats.copy()
    keep_sensors = [
        "Heat_Pump_Heating_Flow_Temperature",
        "Heat_Pump_Return_Temperature",
        "Hot_Water_Flow_Temperature",
    ]
    all_stats = all_stats[all_stats["sensor_type"].isin(keep_sensors)]

    # We expect mean counts per day of 720*2 if all readings are there, if we have less than 50% of the data, we shouldn't use it
    counts = all_stats.groupby("Property_ID")["mean: count per day"].agg("sum").reset_index()
    homes_to_keep = counts.loc[counts["mean: count per day"] >= 0.5 * 720 * 2, "Property_ID"]
    homes_dropped = [file[0:7] for file in files_list if file[0:7] not in homes_to_keep.unique()]

    stats_to_use = all_stats[all_stats["Property_ID"].isin(homes_to_keep)]

    all_stats.loc[
        ~all_stats["Property_ID"].isin(homes_to_keep), "issue"
    ] = "< 50% temperature flow/return readings for home"
    all_stats.loc[~all_stats["Property_ID"].isin(homes_to_keep), "outcome"] = "sensor swaps not checked"

    print(f"Homes kept: {len(homes_to_keep)} ")

    X = stats_to_use.drop(columns=["sensor_type", "Property_ID", "mean: count per day"])
    Y = stats_to_use["sensor_type"] == "Hot_Water_Flow_Temperature"

    n_features = X.shape[1]

    C = 0.002

    # Create classifiers
    classifier_A = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=2,
        min_samples_split=50,
        # min_samples_leaf=100,
        # min_weight_fraction_leaf=0.0,
        # max_features=None,
        random_state=rng,
    )

    classifier_A.fit(X, Y)

    y_pred = classifier_A.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print("Accuracy (train): %0.1f%% " % (accuracy * 100))
    predictions_A = y_pred
    cm = confusion_matrix(Y, predictions_A, labels=classifier_A.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier_A.classes_)
    disp.plot()
    plt.show()

    dot_data2 = StringIO()
    export_graphviz(
        classifier_A,
        out_file=dot_data2,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=X.columns,
        class_names=["Not Hot Water", "Hot Water"],
    )
    graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())
    Image(graph.create_png())

    X = stats_to_use.drop(columns=["sensor_type", "Property_ID"])
    Y = stats_to_use["sensor_type"] == "Hot_Water_Flow_Temperature"

    n_features = X.shape[1]

    C = 0.002

    # Create classifiers
    classifier_B = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=2,
        min_samples_split=50,
        # min_samples_leaf=100,
        # min_weight_fraction_leaf=0.0,
        # max_features=None,
        random_state=rng,
    )

    classifier_B.fit(X, Y)

    y_pred = classifier_B.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print("Accuracy (train): %0.1f%% " % (accuracy * 100))
    predictions_B = y_pred
    cm = confusion_matrix(Y, predictions_B, labels=classifier_B.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier_B.classes_)
    disp.plot()
    plt.show()

    dot_data2 = StringIO()
    export_graphviz(
        classifier_B,
        out_file=dot_data2,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=X.columns,
        class_names=["Not Hot Water", "Hot Water"],
    )
    graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())
    Image(graph.create_png())

    return classifier_A, classifier_B, predictions_A, predictions_B, stats_to_use, all_stats, homes_dropped


def cleaning(
    house_id: str,
    location,
    location_in,
    location_out_cleaned,
    spf_ranges,
    all_dropped_dupes,
    alteration_record,
    all_homes_alteration_record,
    temp_cleaning_type,
    output,
    classifier_A,
    classifier_B,
    file_format,
):
    """Cleans cumulative and non-cumulative columns and saves data

    Args:
        house_id (str): property_id
        location (_type_): files location
        location_in (_type_): files input location
        location_out_cleaned (_type_): location out cleaned location
        spf_ranges (_type_): ranges used for scoring data
        all_dropped_dupes (_type_): dropped duplicate timestamps
        alteration_record (_type_): record of alterations to the data for a single home
        all_homes_alteration_record (_type_): record of alterations for all homes
        temp_cleaning_type (_type_): the group which the determines which temperature cleaning algorithm is used (unflagged, single_flagged, double_flagged, dropped)
        output (_type_): temperature stats with outcome
        classifier_A (_type_): classifier A
        classifier_B (_type_): classifier B

    Returns:
        DataFrame: home_summary_part contains information on the raw/cleaned home
    """

    id = house_id
    file_in_path = os.path.join(location_in, house_id)
    raw_data = utils.load_data(file_in_path, file_format=file_format)

    # Round timestamps of all the raw data and drops duplicates
    formatted_data = qa.round_timestamps(raw_data, n_mins=2)

    # Fix to remove timestamps which are left in trimmed properties when qa.round_timestamps is run after trimming instead of before
    trim_dates_df = pd.read_csv("trim_list.csv")

    if house_id in trim_dates_df:
        trim_date_end = pd.to_datetime(
            trim_dates_df[trim_dates_df["Property_ID"] == house_id].iloc[:, 2], format="%d/%m/%Y"
        ) + pd.Timedelta(days=1)
        formatted_data = formatted_data[~formatted_data.index.isin(trim_date_end)]

    # Drop any duplicated rows (These were introduced by the gap filling)
    if "Timestamp" not in formatted_data.columns:
        formatted_data = formatted_data.reset_index()
    num_rows_start = len(formatted_data)
    formatted_data = formatted_data.drop_duplicates(subset=["Timestamp", "sensor_type"], keep="last")
    dropped_dupes = num_rows_start - len(formatted_data)
    all_dropped_dupes = all_dropped_dupes + dropped_dupes
    formatted_data = formatted_data.set_index("Timestamp")

    data = qa.select_sensors(formatted_data, metric="spfh2")

    # Create home_summary and add Whole_ columns
    home_summary_part = qa.create_home_summary(data, id)
    home_summary_part = home_summary_part.add_prefix("Whole_")
    home_summary_part.rename(columns={"Whole_Property_ID": "Property_ID"}, inplace=True)

    data = data.dropna()

    cleaned_data, alteration_record = ds.correct_reversed_meter(data, alteration_record)
    cleaned_data, alteration_record = ds.remove_anomalous_points(cleaned_data, alteration_record)
    cleaned_data = ds.score_all_data(cleaned_data, spf_ranges)
    cleaned_data, alteration_record = ds.remove_start_with_spf_out_of_range(
        cleaned_data, spf_ranges, alteration_record
    )
    cleaned_data, alteration_record = ds.level_resets(cleaned_data, alteration_record)

    # We want to keep all the alteration records for all homes
    if len(alteration_record) > 0:
        alteration_record["Property_ID"] = id
    all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

    # We want to save all the data out, with the cleaned cumulative data, and non-cleaned temperature data
    temp_sensors = [sensor for sensor in formatted_data["sensor_type"].unique() if sensor.endswith("Temperature")]
    temp_data = formatted_data.loc[formatted_data["sensor_type"].isin(temp_sensors)]
    cleaned_data = cleaned_data.drop("data_score", axis=1)
    # cleaned_data = cleaned_data[["sensor_type", "value", "score", "gap_score", "data_score"]]

    # Add Cleaned_columns to home_summary
    home_summary_part = pd.concat(
        [home_summary_part, qa.create_home_summary(cleaned_data, id).iloc[:, 1:].add_prefix("Cleaned_")], axis=1
    )

    cleaned_data = cleaned_data.dropna()
    output_data = pd.concat([cleaned_data, temp_data])
    # Make sure columns are in correct type
    output_data["sensor_type"] = output_data["sensor_type"].astype(str)
    output_data["value"] = output_data["value"].astype(float)
    output_data.index = pd.to_datetime(output_data.index)

    if temp_cleaning_type == "double_flagged":
        all_homes_alteration_record, cleaned_data = double_flagged_temp_clean(
            output_data,
            house_id,
            output,
            location_out_cleaned,
            classifier_A,
            classifier_B,
            all_homes_alteration_record,
        )
    elif temp_cleaning_type == "single_flagged":
        all_homes_alteration_record, cleaned_data = single_flagged_temp_clean(
            output_data,
            house_id,
            output,
            location_out_cleaned,
            classifier_A,
            classifier_B,
            all_homes_alteration_record,
        )
    elif temp_cleaning_type == "unflagged":
        all_homes_alteration_record, cleaned_data = unflagged_temp_clean(
            output_data, house_id, output, location_out_cleaned, all_homes_alteration_record
        )
    elif temp_cleaning_type == "dropped":
        all_homes_alteration_record, cleaned_data = dropped_temp_clean(
            output_data, house_id, output, location_out_cleaned, all_homes_alteration_record
        )

    # Save cleaned data
    file = house_id
    cleaned_data.to_parquet(
        os.path.join(location_out_cleaned, "cleaned", f"{file}.parquet"), index=False,
    )

    return home_summary_part


def double_flagged_temp_clean(
    output_data, home, output, location_out_cleaned, classifier_A, classifier_B, all_homes_alteration_record
):

    file = home
    id = file.split("=")[-1]
    data = output_data.reset_index()
    temperature_data = utils.filter_temperature_data(data)
    temperature_data = temperature_data.loc[temperature_data["sensor_type"].isin(sensors_may_swap)]

    sensors_to_swap = output.loc[
        (output["Property_ID"] == home) & (output["flagged"]), "sensor_type"
    ].unique()

    mask_A = temperature_data["sensor_type"] == sensors_to_swap[0]
    mask_B = temperature_data["sensor_type"] == sensors_to_swap[1]

    temperature_data.loc[mask_A, "sensor_type"] = sensors_to_swap[1]
    temperature_data.loc[mask_B, "sensor_type"] = sensors_to_swap[0]

    # Check that these sensors characteristics are corrected by the change
    stats = td.get_temperature_stats(temperature_data.reset_index(), id=file.split("=")[-1])

    X = stats.drop(columns=["sensor_type", "Property_ID"])
    prediction = classifier_A.predict(X.drop(columns=["mean: count per day"])) & classifier_B.predict(X)

    # If the changed data is now classified correctly, we consider the change good and output the data
    if (prediction == (stats["sensor_type"] == "Hot_Water_Flow_Temperature")).all():

        # Update the alteration record with the changes made
        alteration_record = ds.add_alteration_record(
            pd.DataFrame(),
            changed_data=temperature_data.loc[mask_A],
            alteration=f"sensor_type changed from {sensors_to_swap[0]}",
            reason="Swapped sensors identified",
        )
        alteration_record = ds.add_alteration_record(
            alteration_record,
            changed_data=temperature_data.loc[mask_B],
            alteration=f"sensor_type changed from {sensors_to_swap[1]}",
            reason="Swapped sensors identified",
        )

        folder = os.path.join(location_out_cleaned, "plots", "double_flagged", "corrected")
        td.plot_data(temperature_data, path=os.path.join(folder, file + ".png"))

        output.loc[output["Property_ID"] == home, "issue"] = ""
        output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
        output.loc[output["Property_ID"] == home, "HWFT full swap"] = True

    else:
        output.loc[output["Property_ID"] == home, "issue"] = "failed: HWFT full swap: not fixed"
        output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"

    # Re-make the full dataset with the temperature sensor data replaced with the swapped version
    data = pd.concat([data.loc[~data["sensor_type"].isin(sensors_may_swap)], temperature_data])

    cleaned_data, alteration_record = td.remove_temperature_anomalies(data, normal_ranges, alteration_record)
    output.loc[output["Property_ID"] == home, "anomalies cleaned"] = True

    alteration_record["Property_ID"] = id
    all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

    return all_homes_alteration_record, cleaned_data


def single_flagged_temp_clean(
    output_data, home, output, location_out_cleaned, classifier_A, classifier_B, all_homes_alteration_record
):

    # For homes where 1 sensor is flagged
    # Many of these appear to be a case where initially the hot water sensor is HP_flow sensors and is then swapped part way through, at which point readings for a missing HP sensor suddenly start.
    # The initial readings appear to combined HP_flow and HW_flow. So data before this point will be flagged

    file = home
    data = output_data.reset_index()
    temperature_data = utils.filter_temperature_data(data)

    # HPHFT and HWFT share a sensor, so we are assuming a single sensor error is with HWFT
    signal = temperature_data.loc[temperature_data["sensor_type"].isin(["Hot_Water_Flow_Temperature"])][
        ["Timestamp", "sensor_type", "value"]
    ]
    signal = (
        signal.dropna()
        .set_index("Timestamp")
        .groupby("sensor_type")
        .resample("1d")["value"]
        .count()
        .dropna()
        .reset_index()
    )

    signal = signal["value"].astype(float).values

    if len(signal) > 0:
        # detection
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=50)

        # display
        rpt.display(signal, [len(signal)], result)
        plt.savefig(
            os.path.join(location_out_cleaned, "plots", "single_flagged", "change_point_analysis", file + ".png",)
        )

        # Result always ends in the number of points in the signal so if it finds 1 change point, result will have length 2
        if len(result) == 2:
            # Result gives number of days at which the signal switches character, at this point we want to
            # retest before and after to see if there is a good split between hot water and not hot water
            start_date = temperature_data.loc[
                temperature_data["sensor_type"] == "Hot_Water_Flow_Temperature", "Timestamp",
            ].dt.date.min()
            change_date = pd.to_datetime(start_date + pd.Timedelta(days=result[0]))
            mask = (temperature_data["Timestamp"] < change_date) & (
                temperature_data["sensor_type"] == "Hot_Water_Flow_Temperature"
            )

            # HPHFT and HWFT share a sensor, so this error should only cause these two sensors to swap
            # We first want to check that HPHFT is empty during the period we plan to swap
            HPHFT_swap_period_empty = (
                len(
                    temperature_data.loc[
                        (temperature_data["sensor_type"] == "Heat_Pump_Heating_Flow_Temperature")
                        & (temperature_data["Timestamp"] < change_date)
                    ]
                )
                == 0
            )

            if HPHFT_swap_period_empty:
                temperature_data.loc[mask, "sensor_type"] = "Heat_Pump_Heating_Flow_Temperature"

                # Check that these sensors characteristics are corrected by the change
                stats = td.get_temperature_stats(temperature_data.copy(), id=file.split("=")[-1])
                X = stats.drop(columns=["sensor_type", "Property_ID"])
                prediction = classifier_A.predict(X.drop(columns=["mean: count per day"])) & classifier_B.predict(X)

                # If the changed data is now classified correctly, we consider the change good and output the data
                if (prediction == (stats["sensor_type"] == "Hot_Water_Flow_Temperature")).all():

                    # Update the alteration record with the changes made
                    alteration_record = ds.add_alteration_record(
                        pd.DataFrame(),
                        changed_data=temperature_data.loc[mask],
                        alteration="sensor_type changed from Hot_Water_Flow_Temperature",
                        reason="Mixed Hot_Water_Flow_Temperature and Heat_Pump_Heating_Flow_Temperature",
                    )

                    folder = os.path.join(location_out_cleaned, "plots", "single_flagged", "corrected")
                    td.plot_data(temperature_data, path=os.path.join(folder, file + ".png"))

                    output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
                    output.loc[output["Property_ID"] == home, "HWFT partial swap"] = True

                else:
                    alteration_record = pd.DataFrame()
                    output.loc[output["Property_ID"] == home, "issue"] = "failed: HWFT partial swap: not fixed"
                    output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
                    output.loc[output["Property_ID"] == home, "HWFT partial swap"] = False
            else:
                alteration_record = pd.DataFrame()
                # The sensor already had data for the swap period so we couldn't do the swap
                # We want to drop the flagged data before the change point
                temperature_data = temperature_data.loc[~mask]
                output.loc[
                    output["Property_ID"] == home, "issue"
                ] = "failed: HWFT partial swap: dropped - no space to swap"
                output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
                output.loc[output["Property_ID"] == home, "HWFT partial swap"] = False
        else:
            alteration_record = pd.DataFrame()
            output.loc[output["Property_ID"] == home, "issue"] = "failed: HWFT partial swap: change point"
            output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
            output.loc[output["Property_ID"] == home, "HWFT partial swap"] = False
    else:
        alteration_record = pd.DataFrame()
        output.loc[output["Property_ID"] == home, "issue"] = "failed: HWFT partial swap: no signal data"
        output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
        output.loc[output["Property_ID"] == home, "HWFT partial swap"] = False

    # Even if we fail to correct the sensor we flagged as unusual we still want to remove anomalies and save all the data out

    # Re-make the full dataset with the temperature sensor data replaced with the swapped version
    data = pd.concat([data.loc[data["sensor_type"].isin(non_temp_sensors)], temperature_data,])

    cleaned_data, alteration_record = td.remove_temperature_anomalies(data, normal_ranges, alteration_record)
    output.loc[output["Property_ID"] == home, "anomalies cleaned"] = True

    alteration_record["Property_ID"] = home
    all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

    return all_homes_alteration_record, cleaned_data


def unflagged_temp_clean(output_data, home, output, location_out_cleaned, all_homes_alteration_record):
    data = output_data

    if len(data) > 0:
        cleaned_data, alteration_record = td.remove_temperature_anomalies(data, normal_ranges, pd.DataFrame())

        output.loc[output["Property_ID"] == home, "issue"] = ""
        output.loc[output["Property_ID"] == home, "outcome"] = "cleaned"
        output.loc[output["Property_ID"] == home, "anomalies cleaned"] = True

        alteration_record["Property_ID"] = home
        all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

        return all_homes_alteration_record, cleaned_data


def dropped_temp_clean(output_data, home, output, location_out_cleaned, all_homes_alteration_record):
    data = output_data

    if len(data) > 0:
        cleaned_data, alteration_record = td.remove_temperature_anomalies(data, normal_ranges, pd.DataFrame())

        output.loc[output["Property_ID"] == home, "issue"] = "< 50% temperature flow/return readings for home"
        output.loc[output["Property_ID"] == home, "outcome"] = "sensor swaps not checked"
        output.loc[output["Property_ID"] == home, "anomalies cleaned"] = True

        alteration_record["Property_ID"] = home
        all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

        return all_homes_alteration_record, cleaned_data
