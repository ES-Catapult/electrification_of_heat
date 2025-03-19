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
# # Explore single home
# This calculates cycling features for a single home and does some visualisations

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import cycling_fns as cf

# %%
# Load in the summary data and filter to ASHPs that were included in the analysis
summary_data = pd.read_csv("DESNZ Electrification of Heat Project - Heat Pump Performance Data Summary.csv")
heat_pumps = summary_data.loc[summary_data["Included_SPF_analysis"], ["Property_ID", "HP_Installed", "HP_Brand", "HP_Model", "HP_Size_kW", "HP_Refrigerant", "SPFH4_selected_window",
                                                                      "HP_Energy_Output_selected_window", "Mean_annual_SH_flow_temp_selected_window", 'Mean_annual_HW_flow_temp_selected_window',
                                                                      'Selected_window_start', 'Selected_window_end']]
heat_pumps = heat_pumps[heat_pumps["HP_Installed"].isin(["ASHP", "HT_ASHP"])]

# %%
# "EOH2232" (cycling temp drop example), "EOH1893" (clear weather comp) "EOH2974" (fixed flow temp) #EOH0311" (two-step fixed weather comp) # "EOH1154" (cycling graph)
property_id = "EOH2232" 
readings = pd.read_csv(f"../../../monitoring_analysis_2/notebooks/clean/Property_ID={property_id}.csv")

# %%
readings = cf.prep_readings(readings)
readings = cf.flag_cycle_groups(readings)
max_power = cf.calc_max_power(readings)
cycles = cf.identify_cycles(readings)
cycling_features = cf.calc_cycling_features(cycles, max_power)

# %%
cycles

# %%
readings.columns

# %%
import plotly.graph_objects as go

# Define the time range
df_filtered = readings.loc['2022-01-08 01:00:00':'2022-01-08 12:00:00'].copy()
df_filtered["comb_flow_temp"] = df_filtered["Heat_Pump_Heating_Flow_Temperature"].fillna(df_filtered["Hot_Water_Flow_Temperature"])

# Create the figure
fig = go.Figure()

# Add the first y-axis (Power values) as bars
fig.add_trace(go.Bar(
    x=df_filtered.index,
    y=df_filtered["Power_W"],
    name="Power (W)",  
    marker_color="black",
    opacity=0.7
))

# fig.add_trace(go.Bar(
#     x=df_filtered.index,
#     y=df_filtered["Heat_Pump_Power_Output"],
#     name="Heat Pump Power Output",
#     marker_color="red",
#     opacity=0.7
# ))

# Add the second y-axis (Flow Temperatures)
fig.add_trace(go.Scatter(
    x=df_filtered.index,
    y=df_filtered["comb_flow_temp"],  # Replace with the actual column name for flow temperature
    name="Flow Temperature (°C)",
    mode="lines",
    line=dict(color="green", width=1),
    yaxis="y2"  # Assign to second y-axis
))

# Add shaded areas where on == 1
for i in range(1, len(df_filtered)):
    if df_filtered['on'].iloc[i] == 1 and df_filtered['on'].iloc[i - 1] == 0:
        # Start of shaded region
        start_time = df_filtered.index[i-1]
    elif df_filtered['on'].iloc[i] == 0 and df_filtered['on'].iloc[i - 1] == 1:
        # End of shaded region
        end_time = df_filtered.index[i]

        if np.isnan(df_filtered["Hot_Water_Flow_Temperature"].iloc[i-1]):
            fill_colour="rgba(255, 0, 0, 0.3)"
        else:
            fill_colour="rgba(0, 0, 255, 0.3)"
        # Add shaded area
        fig.add_shape(
            type="rect",
            x0=start_time,
            x1=end_time,
            y0=0, y1=1,  # Extend full height
            xref="x",
            yref="paper",
            fillcolor=fill_colour,
            line_width=0
        )

fig.add_shape(
    type="line",
    x0=df_filtered.index.min(),
    x1=df_filtered.index.max(),
    y0=200,
    y1=200,
    line=dict(color="black", width=2, dash="dash"),  # Customize color, width, and dash style
)


# Update layout to include secondary y-axis
fig.update_layout(
    width=900,
    title="Detecting Heating and Hot Water Cycles",
    xaxis=dict(title="Time"),
    yaxis=dict(
        title="Heat Pump Electrical Power (W)",
        side="left",
        showgrid=False
    ),
    yaxis2=dict(
        title="Flow Temperature (°C)",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    legend=dict(x=0, y=1)    
)

# Show the figure
fig.show()


# %%
readings['2022-01-08 04:50':'2022-01-08 05:20']

# %%
px.scatter(cycles[(cycles["state"]==1) & ~cycles["hot_water"]], x="mean_external_temp", y="max_heat_temp", 
           labels={
                "mean_external_temp": "External Temperature (C)",
                "max_heat_temp": "Maximum Heating Flow Temperature (C)",
            }, width=600)

# %%
fig = px.scatter(cycles[(cycles["state"]==1) & ~cycles["hot_water"]], x="mean_external_temp", y="median_heat_temp", 
           title=f"Empirical weather compensation curve for {property_id}",
           labels={
                "mean_external_temp": "External Temperature (C)",
                "median_heat_temp": "Median Heating Flow Temperature (C)",
            },           
           width=600,
           trendline="ols")
# Change trendline color
#fig.update_traces(selector=dict(name="OLS Trendline"), line=dict(color="black", width=3))

# Force update of the trendline color
fig.data[-1].line.color = "black"  # Trendline is usually the last trace
fig.data[-1].line.width = 3  # Make it more visible
fig.show()

# %%
px.scatter(cycles[(cycles["state"]==1) & ~cycles["hot_water"]], x="start_time", y="median_heat_temp", width=600)

# %%
readings.columns

# %%
