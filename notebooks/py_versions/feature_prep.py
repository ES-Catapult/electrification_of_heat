# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: EoH
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cycling Features Preparation
# This calculates cycling features for all the ASHPs included in the analysis and outputs that to homes_with_features.csv
#
# Note: Updated to use a power_floor of 200W due to seeing properties like EOH1446 which was creating cycles from 'noise' around the 100W mark, and EOH2074 which wasn't having cycles split out at 100W.

# %%
import pandas as pd
import cycling_fns as cf
from tqdm.notebook import tqdm

# %%
# Load in the summary data and filter to ASHPs that were included in the analysis
summary_data = pd.read_csv("DESNZ Electrification of Heat Project - Heat Pump Performance Data Summary.csv")
heat_pumps = summary_data.loc[summary_data["Included_SPF_analysis"], ["Property_ID", "HP_Installed", "HP_Brand", "HP_Model", "HP_Size_kW", "HP_Refrigerant", 
                                                                      "SPFH4_selected_window", "HP_Energy_Output_selected_window", "Mean_annual_SH_flow_temp_selected_window", 'Mean_annual_HW_flow_temp_selected_window',
                                                                      'Selected_window_start', 'Selected_window_end',]]
heat_pumps = heat_pumps[heat_pumps["HP_Installed"].isin(["ASHP", "HT_ASHP"])]

# %%
# Get additional installation data
install_data = pd.read_csv("BEIS Electrification of Heat Project - Property, Design and Installation Information.csv")
install_data = install_data[["Property_ID", "Delivery_Contractor", "Name_Survey", "Name_Install", 
                             "MCS_Flow_Temp", "MCS_SCOP","MCS_SHLoad", "MCS_Hloss", "MCS_SHAnnual", "MCS_DHWAnnual"]]

# %%
heat_pumps = heat_pumps.merge(install_data, on="Property_ID")

# %%
# Loop over all 2-minute data from the properties and calculate cycling features for them
homes_with_features = []

for index, row in tqdm(heat_pumps.iterrows(), total=len(heat_pumps)):
    property_id = row.Property_ID
    readings = pd.read_csv(f"../../../monitoring_analysis_2/notebooks/clean/Property_ID={property_id}.csv")
    try:
        cycling_features = cf.get_all_features(readings, power_floor=200)
        annual_features = cf.get_annual_features(readings, pd.Timestamp(row.Selected_window_start), pd.Timestamp(row.Selected_window_end))
    except:
        print(f"Unable to get features for {property_id}")

    homes_with_features.append(pd.concat([row, cycling_features, annual_features]))

homes_with_features = pd.DataFrame(homes_with_features)

# %%
homes_with_features.to_csv("homes_with_features.csv", index=False)

# %%
