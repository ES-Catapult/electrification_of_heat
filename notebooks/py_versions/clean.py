# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.10.8 64-bit (windows store)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import os
import qa_functions as qa
import utils
import temperature_data_fns as td
import numpy as np
import matplotlib.pyplot as plt
import cleaning_fns as clf
from tqdm import tqdm

pd.set_option("display.max_columns", None)

# %%
# Set file format for input files, either parquet or csv
file_format = qa.set_file_format(file_format="parquet")

# %%
# Specify which folder contains your local directory
eoh_folder = os.environ.get("EoH")

# Specify location of specific folders
location = os.path.join(eoh_folder, "EOH_Data_Local")
location_in_raw = os.path.join(location, "raw")
location_out = os.path.join(location, "processed")
location_out_cleaned = os.path.join(location_out, "cleaned")
files_list = os.listdir(location_in_raw)

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
# # This code is commented out as it takes some time to run
# # You can run the code in this cell to generate the temperature_stats.csv
# # Alternately if you have already generated the csv then
# # you can load the csv in the following cell to view the plots below

# file_names = os.listdir(location_in_raw)
# all_stats = pd.DataFrame()
# for file in file_names:
#     print(file)
#     id = file.split("=")[-1]
#     temperature_data = utils.load_temperature_data(os.path.join(location_in_raw_2, file), file_format=file_format)
#     temperature_data = qa.round_timestamps(temperature_data, n_mins=2)

#     # path = os.path.join(location_out, "plots", "temperature", file + ".png")
#     # td.plot_data( temperature_data, path=path )
#     temperature_data = temperature_data.reset_index()
#     stats = td.get_temperature_stats(temperature_data, id)
#     all_stats = pd.concat([all_stats, stats])

# path = os.path.join(location_out, "temperature_stats.csv")
# all_stats.to_csv(path, index=False)
# # all_stats.to_csv(path, mode="a", index=False, header=False)

# %%
# Using the temperature_stats.csv file, we create classifiers and make predictions
# and create the homes_dropped subset of homes
(
    classifier_A,
    classifier_B,
    predictions_A,
    predictions_B,
    stats_to_use,
    all_stats,
    homes_dropped,
) = clf.classify_and_predict(location_out, files_list)

# %%
# The code in this cell generates 3 more subsets of homes:
# unflagged homes, single_flagged_homes, double_flagged homes

output = stats_to_use.copy()
output["prediction_A"] = predictions_A
output["prediction_B"] = predictions_B
output["flagged_A"] = output["prediction_A"] != (output["sensor_type"] == "Hot_Water_Flow_Temperature")
output["flagged_B"] = output["prediction_B"] != (output["sensor_type"] == "Hot_Water_Flow_Temperature")

# There is a fair difference between how the splits behave, we will have a look at all of them
output["flagged"] = output["flagged_A"] | output["flagged_B"]

home_flags = output.groupby(["Property_ID"])["flagged"].sum()
output = pd.merge(output, home_flags, on="Property_ID", how="left", suffixes=[None, "_per_home"])

# Merge in all_stats to get the issue for the homes dropped for having < 50% temp data
output = pd.merge(
    output,
    all_stats[["sensor_type", "Property_ID", "issue"]],
    on=["sensor_type", "Property_ID"],
    how="outer",
)

flagged_homes = output[output["flagged_per_home"] > 0]
flagged_homes = flagged_homes.groupby("Property_ID").agg({"flagged_per_home": "max"})
unflagged_homes = np.sort(output.loc[output["flagged_per_home"] == 0]["Property_ID"].unique())
single_flagged_homes = flagged_homes.loc[flagged_homes["flagged_per_home"] == 1]
double_flagged_homes = flagged_homes.loc[flagged_homes["flagged_per_home"] == 2]
single_flagged_homes = single_flagged_homes.index.sort_values()
double_flagged_homes = double_flagged_homes.index.sort_values()

# %%
# This cell is where cleaning for each home takes

home_classifications = {
    "unflagged": unflagged_homes,
    "single_flagged": single_flagged_homes,
    "double_flagged": double_flagged_homes,
    "dropped": homes_dropped,
}

# Loop through each home
all_dropped_dupes = 0
window_method = "best"
homes_with_error = []
home_summary = []
all_windows = pd.DataFrame()
all_homes_alteration_record = pd.DataFrame()
alteration_record = pd.DataFrame()

for home_classification in ["unflagged", "single_flagged", "double_flagged", "dropped"]:
    for i, home in zip(
        tqdm(range(len(home_classifications.get(home_classification)))), home_classifications.get(home_classification)
    ):
        temp_cleaning_type = home_classification

        home_summary_part = clf.cleaning(
            home,
            location,
            location_in_raw,
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
        )

        home_summary.append(home_summary_part)

home_summary = pd.concat(home_summary)
home_summary

# %%
# Save the outcomes output
output = output.sort_values("Property_ID")
output.to_csv(os.path.join(location_out, "temperature_stats_with_outcome.csv"))

# %%
# Save home summary
home_summary.to_csv(os.path.join(location_out, "home_summary_partial_1.csv"), index=False)

# %%
print(f"Total dropped duplicates: {all_dropped_dupes}")

# Save out the alteration record for all the homes
all_homes_alteration_record.to_csv(os.path.join(location_out, "summary_of_alterations.csv"), index=False)
