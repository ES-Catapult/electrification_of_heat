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
# # Creating heatmaps of modulation and flow temperature
# This creates a heatmap of how much time each of the ASHPs in the analysis spends in different combinations of flow temperature and power modulation. This might be useful for quick visual diagnosis of systems. Currently I'm not too confident the ceiling of the modulation is correct, so improving that is a priority (see below for suggestions).
#
# ### Rough guide to diagnosis
# This is my best guess at what different shapes might represent - but needs validation and improvement.
# - The best performing are probably going to be high density in the bottom left combined with long median cycles and short gaps. This corresponds to a system where the weather compensation and controls have been really well calibrated to produce continuous heating at low flow temperatures.
# - Bottom left but with short cycles and gaps and lower SPFH4 probably corresponds to a system that is trying to operate below the minimum possible modulation. Raising the minimum flow temperature will help achieve longer cycles, but the system is probably oversized and swapping in a smaller heat pump might lead to significant performance improvements.
# - A strong horizontal line in the middle or top may correspond to a property set to a fixed flow temperature. Weather compensation should be turned on.
# - High density on the right with long cycles and short gaps would likely indicate an undersized heat pump that is struggling to deliver the required heat output.
# - ??? What would the weather compensation curve being set too high look like? High density towards the middle and top and shorter cycles?
#
# ### Potential improvements
# - A lookup table of max powers by model would be make these figures more reliable, rather than relying on deriving a rough max power from each unit. This could be sourced from manufacturers or something added into the calc_cycling_features() function and then a lookup table generated from homes_with_features.csv.
# - Ideally the minimum modulation we'd expect to be possible should be marked (e.g. with a vertical line). This should take into account where different models are just software limited versions of the same hardware so that the true range of the compressor is reflected.
# - Basing the colour on Whole_System_Energy_Consumed rather than number of 2-minute intervals might show a slightly different story that more directly correlates with performance. I didn't do that initially because that naturally gets weighted to the right, which I wasn't sure was desirable.
#
# ### Alternative heatmaps
# Here are some suggestions for alternative heatmaps that might be equally/more useful. Looking at the cycles themselves and plotting:
# - on_duration vs off_duration
# - modulation vs on_duration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cycling_fns as cf
from tqdm.notebook import tqdm

# %%
data_path = r"C:\Users\samuel.young\OneDrive - Energy Systems Catapult Ltd\Projects\EoH\monitoring_analysis_2\notebooks\clean"

# %%
# Load in the summary data that includes cycling features
heat_pumps = pd.read_csv("homes_with_features.csv")

# %%
power_floor_w = 100 # Discard heat pump power input readings below this
power_ceiling_w = 5000 # Discard heat pump power input readings above this
max_power_percentile = 95 # Used to derive the empirical maximum rated power. 99 or 95 are probably good options (to remove outliers)

# %%
# Loop over the 2 minute data for each heat pump
for index, row in tqdm(heat_pumps.iterrows(), total=len(heat_pumps)):
    property_id = row.Property_ID
    # Create the figure name. Note we start with SPFH to make it easier to sort them by performance and look for patterns
    figure_name = f"../figures/{row.SPFH4_selected_window:.2f} - {property_id}.png"
    if os.path.exists(figure_name):
        print(f"{property_id} already processed")
        continue

    try:
        readings = pd.read_csv(os.path.join(data_path, f"Property_ID={property_id}.csv"))

        readings = cf.prep_readings(readings, power_floor=power_floor_w)
        
        # Remove unreasonably high or low power readings
        sensible_power = readings.loc[(readings["Power_W"] >= power_floor_w) & (readings["Power_W"] <= power_ceiling_w)].copy()

        # Calculate the maximum power and modulation
        max_power = np.percentile(sensible_power["Power_W"], max_power_percentile).round(-1)
        sensible_power["Modulation %"] = sensible_power["Power_W"] / max_power * 100

        # Only look at periods where we have valid data and hot water is not being used
        sensible_power = sensible_power[(sensible_power["Heat_Pump_Energy_Output"].notna()) 
                                        & sensible_power["External_Air_Temperature"].notna()
                                        & sensible_power["Heat_Pump_Heating_Flow_Temperature"].notna()
                                        & sensible_power["Modulation %"].notna()
                                        & sensible_power["Hot_Water_Flow_Temperature"].isna()]
        
        sensible_power["Modulation % Rounded"] = ((sensible_power["Modulation %"] / 5).round(0) * 5).astype(int)
        sensible_power["Heat_Pump_Heating_Flow_Temperature_Rounded"] = sensible_power["Heat_Pump_Heating_Flow_Temperature"].round(0).astype(int)
        heatmap_data = sensible_power.groupby(['Heat_Pump_Heating_Flow_Temperature_Rounded', 'Modulation % Rounded']).size().unstack()

        # Filter to between 25 and 65C
        full_index = pd.Index(range(65, 24, -1), name="Heat_Pump_Heating_Flow_Temperature_Rounded")
        full_columns = pd.Index(range(5, 101, 5), name="Modulation % Rounded")
        heatmap_data = heatmap_data.reindex(index=full_index, columns=full_columns, fill_value=0)
        heatmap_data = heatmap_data.fillna(0)

        # Plot heatmap
        plt.figure(figsize=(7, 6))
        sns.heatmap(heatmap_data, cmap='inferno', annot=False, fmt='g', cbar=False)
        plt.yticks(rotation=0)
        plt.xlabel(f'Modulation (as % of calculated max power: {max_power/1000}kW)')
        plt.ylabel('Flow Temperature (C)')
        run_time = int(round(len(sensible_power)/30,0))
        plt.title((f"Modulation Diagnostic for {property_id} (SPFH4={row.SPFH4_selected_window:.2f})\n"
                  f"{row.HP_Size_kW}kW {row.HP_Model} \n "
                  f"Colour is % of winter run time ({run_time}h) spent in that state"))
        plt.text(0.1, 1.5, f"Median cycle: {row.median_on_duration}m", color="white", fontsize=12, ha='left')
        plt.text(0.1, 3, f"Median gap: {row.median_off_duration}m", color="white", fontsize=12, ha='left')

        plt.savefig(figure_name, dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print(f"Error processing {property_id}")
        continue
    

# %%
