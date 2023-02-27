# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.10.8 64-bit (microsoft store)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import os
import qa_functions as qa
from tqdm import tqdm
import temperature_data_fns as td

pd.set_option("display.max_columns", None)

# %%
# Set file format
file_format = qa.set_file_format(file_format="parquet")

# %%
# Specify which folder contains your local directory
eoh_folder = os.environ.get("EoH")

# Specify read and write locations
location = os.path.join(eoh_folder, "EOH_Data_Local")
location_out = os.path.join(location, "processed")
location_out_cleaned = os.path.join(location_out, "cleaned")
downloaded = os.listdir(os.path.join(location_out_cleaned, "cleaned"))

# %%
# Define the boundaries for the ranges of gap length
# duration < short - Not a gap
# short < duration < medium - a short gap
# medium < duration < long - a medium gap
# duration > long - a long gap
gap_len_defs = {
    "long": pd.Timedelta(days=21),
    "medium": pd.Timedelta(days=7),
    "short": pd.Timedelta(minutes=30),
}

# %%
# Define the performance factor ranges for different time scales
# For short time scales, we expect higher variation in the performance factor
# short time periods are around a day, long are around a year, medium is in between these two
spf_ranges = {
    "short": {"min": 0.75, "max": 7.5},
    "medium": {"min": 0.9, "max": 6.5},
    "long": {"min": 1.5, "max": 5.0},
}

# %%
window_method = "best"
save_scored_data = True
plotting = True
all_windows = pd.DataFrame()
home_summary = []
all_homes_alteration_record = pd.DataFrame()
home_summary_cleaned = pd.read_csv(os.path.join(location, "processed", "home_summary_partial_1.csv"))
for i, home in zip(tqdm(range(len(downloaded))), downloaded):
    # print(home)
    id = home.split("=")[1].split("\\")[0]
    if plotting:
        # Un-comment to save the plots
        plot_full_save_path = os.path.join(location_out_cleaned, "plots", "full", home + ".png")
        plot_window_save_path = os.path.join(location_out_cleaned, "plots", "window", home + ".png")
        # Un-comment for fig.show()
        # plot_full_save_path = "none"
        # plot_window_save_path = "none"
    else:
        plot_full_save_path = ""
        plot_window_save_path = ""

    if file_format == "csv":
        data = pd.read_csv(os.path.join(location_out_cleaned, "cleaned", home)).set_index("Timestamp")
    elif file_format == "parquet":
        data = pd.read_parquet(os.path.join(location_out_cleaned, "cleaned", home)).set_index("Timestamp")

    home_summary_part = home_summary_cleaned[home_summary_cleaned["Property_ID"] == id]

    # Select the best window for this data, this function also adds stats about the window to the output
    (window_data, home_summary_part, cleaned_data, windows, single_home_alteration_record,) = qa.select_window(
        data,
        gap_len_defs,
        home_summary_part,
        spf_ranges=spf_ranges,
        window_len_mths=12,
        method=window_method,
        plot_full_save_path=plot_full_save_path,
        plot_window_save_path=plot_window_save_path,
        location_out_cleaned=location_out_cleaned,
        file=home,
        save_scored_data=save_scored_data,
    )

    # Add most common flow temperatures to home_summary
    file_path = os.path.join(location_out, "binned_heating_temperature", home + ".csv")
    plot_path = os.path.join(location_out, "binned_heating_temperature", "plots", home + ".png")

    # We only want to add these stats if a window exists
    if "window_start" in home_summary_part.columns:
        home_summary_part = td.add_flow_temp_stats_for_window(
            data, home_summary_part, id, file_path=file_path, plot_path=plot_path
        )

    # Find spfs for coldest day
    home_summary_part_cold_day = td.add_spfs_for_coldest_period(data, id, pd.Timedelta(days=1), "Coldest_day_")

    # Find spfs for coldest half-hour
    home_summary_part_cold_HH = td.add_spfs_for_coldest_period(data, id, pd.Timedelta(minutes=30), "Coldest_HH_")

    if (len(home_summary_part_cold_day) > 0) & (len(home_summary_part_cold_HH) > 0):
        home_summary_part_cold = pd.merge(
            home_summary_part_cold_day,
            home_summary_part_cold_HH,
            on="Property_ID",
            how="outer",
        )

        home_summary_part = home_summary_part.merge(home_summary_part_cold, on="Property_ID")

    home_summary_part["window_method"] = window_method
    home_summary.append(home_summary_part)

    # We want to save all the possible windows out separately to do some analysis on them
    windows["Property_ID"] = home.split("=")[1]
    all_windows = pd.concat([all_windows, windows], axis=0)

    # We want to keep all the alteration records for all homes
    if len(single_home_alteration_record) > 0:
        single_home_alteration_record["Property_ID"] = id
    all_homes_alteration_record = pd.concat([all_homes_alteration_record, single_home_alteration_record])

home_summary = pd.concat(home_summary)

cleaning_flags = pd.read_csv(os.path.join(location_out, "temperature_stats_with_outcome.csv"))
cleaning_flags = (
    cleaning_flags.groupby("Property_ID")
    .max()[["issue", "outcome", "anomalies cleaned", "HWFT partial swap", "HWFT full swap"]]
    .add_prefix("temperature_cleaning_")
    .reset_index()
)
cleaning_flags["Property_ID"] = cleaning_flags["Property_ID"].astype(str)

home_summary = pd.merge(home_summary, cleaning_flags, on="Property_ID")

home_summary

# %%
# If running the whole set of homes, we want to over-write the old file
home_summary.to_csv(os.path.join(location_out, "home_summary.csv"), index=False)

# save out all the windows data to file
all_windows.to_csv(os.path.join(location_out, "all_windows.csv"), index=False)

# %%
# We save a redacted version of the home summary file to allow a quick comparison to the output of previous code versions
home_summary_partial_2 = home_summary[
    [
        "Property_ID",
        "Whole_start",
        "Whole_end",
        "Whole_duration_days",
        "Whole_%_complete_Circulation_Pump_Energy_Consumed",
        "Whole_%_complete_Heat_Pump_Energy_Consumed",
        "Whole_%_complete_Heat_Pump_Energy_Output",
        "Whole_%_complete_Immersion_Heater_Energy_Consumed",
        "Cleaned_start",
        "Cleaned_end",
        "Cleaned_duration_days",
        "Cleaned_%_complete_Circulation_Pump_Energy_Consumed",
        "Cleaned_%_complete_Heat_Pump_Energy_Consumed",
        "Cleaned_%_complete_Heat_Pump_Energy_Output",
        "Cleaned_%_complete_Immersion_Heater_Energy_Consumed",
        "acceptable windows: spfh2: count",
        "acceptable windows: spfh2: mean",
        "acceptable windows: spfh2: std",
        "acceptable windows: spfh2: min",
        "acceptable windows: spfh2: 25%",
        "acceptable windows: spfh2: 50%",
        "acceptable windows: spfh2: 75%",
        "acceptable windows: spfh2: max",
        "acceptable windows: spfh3: count",
        "acceptable windows: spfh3: mean",
        "acceptable windows: spfh3: std",
        "acceptable windows: spfh3: min",
        "acceptable windows: spfh3: 25%",
        "acceptable windows: spfh3: 50%",
        "acceptable windows: spfh3: 75%",
        "acceptable windows: spfh3: max",
        "acceptable windows: spfh4: count",
        "acceptable windows: spfh4: mean",
        "acceptable windows: spfh4: std",
        "acceptable windows: spfh4: min",
        "acceptable windows: spfh4: 25%",
        "acceptable windows: spfh4: 50%",
        "acceptable windows: spfh4: 75%",
        "acceptable windows: spfh4: max",
        "window_start",
        "window_end",
        "window_max_gap_score",
        "window_max_data_score",
        "window_max_score",
        "window_mean_gap_score",
        "window_mean_data_score",
        "window_mean_score",
        "window_Circulation_Pump_Energy_Consumed",
        "window_Heat_Pump_Energy_Consumed",
        "window_Heat_Pump_Energy_Output",
        "window_Immersion_Heater_Energy_Consumed",
        "window_Back-up_Heater_Energy_Consumed",
        "window_Boiler_Energy_Output",
        "spfh2",
        "spfh3",
        "spfh4",
        "window_duration_days",
        "window_%_complete_Circulation_Pump_Energy_Consumed",
        "window_%_complete_Heat_Pump_Energy_Consumed",
        "window_%_complete_Heat_Pump_Energy_Output",
        "window_%_complete_Immersion_Heater_Energy_Consumed",
        "window_method",
        "Whole_%_complete_Boiler_Energy_Output",
        "Cleaned_%_complete_Boiler_Energy_Output",
        "window_%_complete_Boiler_Energy_Output",
        "Whole_%_complete_Back-up_Heater_Energy_Consumed",
        "Cleaned_%_complete_Back-up_Heater_Energy_Consumed",
        "window_%_complete_Back-up_Heater_Energy_Consumed",
    ]
]
home_summary_partial_2.to_csv(os.path.join(location_out, "home_summary_partial_2.csv"), index=False)
