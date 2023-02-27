#!/usr/bin/env python
# coding: utf-8

# ## BEIS Electrification of Heat Project - Analysis

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import chart_functions as cf
import analysis_functions as af
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pylab
import numpy as np
import os

pio.renderers.default = "notebook"

# Set no limit to the number of columns displayed
pd.set_option("display.max_columns", None)

# Set plotly colour scheme
pio.templates["EoH"] = go.layout.Template(
    layout_colorway=["#%02x%02x%02x" % rgb_colour for rgb_colour in cf.colour_scheme().values()]
)
pio.templates.default = "EoH"

# Set opacity of lines and markers on scatter plots to allow high density scatters to display better
scatter_opacity = 0.7


# In[ ]:


# Specify location for read and write
eoh_folder = os.environ.get("EoH")


# In[ ]:


# Read in data, 1 row per home
home_summary = pd.read_csv(os.path.join(eoh_folder, "EOH_Data_Local", "processed", "home_summary.csv"))

# Add supplementary data columns from usmart
home_summary = af.add_supplementary_data(home_summary)


# #### Data Selection
# 742 homes were installed with a heat pump; a number drop out of the analysis for a variety of reasons.
# 
# An algorithm was devised to select a 12-month window of data (out of the entire date range) for each home, prioritising higher quality data.
# 
# The waterfall chart below provides detail and commentary on why and what homes are dropped.
# - SPF values outside of the range 1.5<SPF_Value<4.5 are deemed to be erroneous as SPFs outside that range are highly unlikely from a physical perspective.
# - An algorithm was devised to score the quality of data from cumulative meters received for each home. This focused on assessing how confident we are that the meter incremented continuously over the entire period (particularly during gaps in the data feed). A window_max_gap_score score of 4 or above is deemed to be unreliable, as this indicates either extremely long gaps with no apparent energy use, or a decrease in the meter reading during a long gap.

# In[ ]:


# Set analysis thresholds
SPF_value_min_threshold = 1.5
SPF_value_max_threshold = 4.5
window_max_score_threshold = 4
still_monitoring_cut_off = pd.to_datetime("2022-08-01", format="%Y-%m-%d")

# Create homes inclusion waterfall chart (v2) and adds exclusion fields to DataFrame
home_summary = cf.chart_homes_waterfall2(
    home_summary,
    SPF_value_min_threshold,
    SPF_value_max_threshold,
    window_max_score_threshold,
    still_monitoring_cut_off,
)

# Save copy of the home_summary file with supplementary columns
home_summary.to_csv(os.path.join(eoh_folder, "EOH_data_local", "processed", "home_summary_with_install_info.csv"), index=False)


# In[ ]:


# Create homes inclusion waterfall chart and create exclusion fields
cf.chart_homes_waterfall(
    home_summary,
    SPF_value_min_threshold,
    SPF_value_max_threshold,
    window_max_score_threshold,
    still_monitoring_cut_off,
)


# In[ ]:


# Prepare the data for analysis

# Reformat to 1 row per home per SPF_type
home_summary = af.reformat_SPF_to_long(home_summary, cold_analysis=True)

# Calculate total energy output
home_summary["window_Total_Energy_Output"] = (
    home_summary["window_Heat_Pump_Energy_Output"].fillna(0)
    + home_summary["window_Boiler_Energy_Output"].fillna(0)
    + home_summary["window_Immersion_Heater_Energy_Consumed"].fillna(0)
    + home_summary["window_Back-up_Heater_Energy_Consumed"].fillna(0)
)

# Split DataFrame into two: SPF and COP calculations
home_summary_cold = home_summary[
    (home_summary["SPF_type"] != "SPFH2")
    & (home_summary["SPF_type"] != "SPFH3")
    & (home_summary["SPF_type"] != "SPFH4")
]
home_summary = home_summary[
    (home_summary["SPF_type"] == "SPFH2")
    | (home_summary["SPF_type"] == "SPFH3")
    | (home_summary["SPF_type"] == "SPFH4")
]

# Filter DataFrame to just those homes that are within the thresholds
homes_within_threshold = home_summary[home_summary["Removed_from_analysis"] == False]


# In[ ]:


# Show performance for GSHPs before removing
pd.DataFrame(round(
    homes_within_threshold[homes_within_threshold["HP_Type"] == "GSHP"]
    .groupby(["Property_ID", "SPF_type"])["SPF_value"]
    .first(),2)
).reset_index().pivot(index="Property_ID", columns="SPF_type", values="SPF_value")


# In[ ]:


# Remove GSHPs from analysis
homes_within_threshold = homes_within_threshold[
    homes_within_threshold["HP_Type"] != "GSHP"
]


# In[ ]:


# Count homes per delivery contractor
pd.DataFrame(homes_within_threshold.groupby("Delivery_Contractor")["Property_ID"].nunique())


# In[ ]:


# # Optional code to impute erroneous/missing circulation pump data

# Print the number of homes with a valid circ pump value
valid_circs = homes_within_threshold[homes_within_threshold["window_Circulation_Pump_Energy_Consumed"] > 0][
    "window_Circulation_Pump_Energy_Consumed"
].size
print(f"Valid circulation pump homes: {valid_circs/3}")

# Calculate the mean circ pump value for valid circ pump homes
mean_circ = homes_within_threshold[homes_within_threshold["window_Circulation_Pump_Energy_Consumed"] > 0][
    "window_Circulation_Pump_Energy_Consumed"
].mean()
print(f"Mean circulation pump value for valid homes: {round(mean_circ,1)}")
print("")
for SPF_type in ["SPFH2", "SPFH3"]:
    print(SPF_type)
    df = homes_within_threshold[homes_within_threshold["SPF_type"] == SPF_type].copy()
    # Generate stats on homes that qualify for the threshold
    print("Original data")
    SPF_stats = af.stats_by_hp_and_SPF_type(df, by="HP_Type_3", rounding=3)[
        ["HP_Type_3", "SPF_type", "Homes", "Median", "Median (IQR)", "Mean (95% CI)"]
    ].set_index(["HP_Type_3", "SPF_type"])
    display(SPF_stats)

    # Impute mean circ pump value into homes with invalid circ pump values
    df["window_Circulation_Pump_Energy_Consumed"] = np.where(
        df["window_Circulation_Pump_Energy_Consumed"] == 0,
        mean_circ,
        df["window_Circulation_Pump_Energy_Consumed"],
    )

    # Recalculate SPF
    if SPF_type == "SPFH2":

        df["SPF_value"] = df["window_Heat_Pump_Energy_Output"] / (
            df["window_Whole_System_Energy_Consumed"]
            - df["window_Circulation_Pump_Energy_Consumed"]
            - df["window_Immersion_Heater_Energy_Consumed"]
            - df["window_Back-up_Heater_Energy_Consumed"]
        )

    elif SPF_type == "SPFH3":

        df["SPF_value"] = (
            df["window_Heat_Pump_Energy_Output"]
            + df["window_Immersion_Heater_Energy_Consumed"]
            + df["window_Back-up_Heater_Energy_Consumed"]
        ) / (df["window_Whole_System_Energy_Consumed"] - df["window_Circulation_Pump_Energy_Consumed"])

    # Generate stats on homes that qualify for the threshold
    print("Edited data")
    SPF_stats_edited = af.stats_by_hp_and_SPF_type(df, by="HP_Type_3", rounding=3)[
        ["HP_Type_3", "SPF_type", "Homes", "Median", "Median (IQR)", "Mean (95% CI)"]
    ].set_index(["HP_Type_3", "SPF_type"])
    display(SPF_stats_edited)
    edited_median = SPF_stats_edited["Median"]
    median = SPF_stats["Median"]
    print(f"{SPF_type} difference (median): {round(edited_median - median,3)[0]}")
    print("")
    print("")
    print("")


# #### Headline SPF Statistics
# The tables below show the headline SPF statistics for each heat pump type in the trial. The table is split by HP_Type, HP_Type_2 and HP_Type_3.
# 
# HP_Type_2 groups LT_ASHPs and HT_ASHPs together into 'ASHPs'.
# 
# HP_Type_3 groups hybrids and ASHPs together into 'All_ASHPs'.
# 
# GSHPs have been removed from the dataset as there are currently only two units with enough data for SPF to be calculated. These GSHPs have SPFH2 values of 2.71 and 4.2, and SPFH4 values of 2.46 and 3.8.

# In[ ]:


# Optional code

# Generate stats on homes that qualify for the threshold
af.stats_by_hp_and_SPF_type(homes_within_threshold, rounding=2)[
    ["HP_Type", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]
].set_index(["HP_Type", "SPF_type"]).sort_values(["HP_Type", "SPF_type"])


# ASHPs have a median SPFH4 value of 2.82, Hybrids have a median SPFH4 value of 2.46.

# In[ ]:


homes_within_threshold = homes_within_threshold.sort_values("SPF_type")


# In[ ]:


# Generate stats on homes that qualify for the threshold
af.stats_by_hp_and_SPF_type(homes_within_threshold, by="HP_Type_2")[
    ["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]
].set_index(["HP_Type_2", "SPF_type"]).sort_values(["HP_Type_2", "SPF_type"])


# In[ ]:


box_plot_df = homes_within_threshold[
    ~(
        (homes_within_threshold["SPF_type"] == "SPFH3")
        & (homes_within_threshold["HP_Type"] == "Hybrid")
    )
].sort_values("SPF_type")

px.box(
    box_plot_df,
    x="HP_Type_2",
    y="SPF_value",
    color="SPF_type",
    width=700,
    labels={"HP_Type_2": "Heat Pump Type", "SPF_value": "SPF"},
)


# In[ ]:


box_plot_df_2 = homes_within_threshold[homes_within_threshold["HP_Type_2"] == "ASHPs"]

px.box(
    box_plot_df_2,
    x="HP_Type",
    y="SPF_value",
    color="SPF_type",
    width=700,
    labels={"HP_Type_2": "Heat Pump Type", "SPF_value": "SPF"},
)


# In[ ]:


# t-test across all SPF types for LT and HT ASHPs

lt_ashps = homes_within_threshold[(homes_within_threshold["HP_Type"] == "LT_ASHP")]["SPF_value"]
ht_ashps = homes_within_threshold[(homes_within_threshold["HP_Type"] == "HT_ASHP")]["SPF_value"]

statistic, pvalue = stats.ttest_ind(lt_ashps, ht_ashps)
if pvalue < 0.05:
    print("Significant difference in means between LT and HT ASHPs across all SPF types")
else:
    print("No significant difference in means between LT and HT ASHPs across all SPF types")


# In[ ]:


# t-test on just SPFH4 for LT and HT ASHPs

lt_ashps = homes_within_threshold[
    (homes_within_threshold["HP_Type"] == "LT_ASHP") & (homes_within_threshold["SPF_type"] == "SPFH4")
]["SPF_value"]
ht_ashps = homes_within_threshold[
    (homes_within_threshold["HP_Type"] == "HT_ASHP") & (homes_within_threshold["SPF_type"] == "SPFH4")
]["SPF_value"]

statistic, pvalue = stats.ttest_ind(lt_ashps, ht_ashps)
if pvalue < 0.05:
    print("Significant difference in means between LT and HT ASHPs across SPFH4")
else:
    print("No significant difference in means between LT and HT ASHPs across SPFH4")


# In[ ]:


# t-test across all SPF types for ASHPs and Hybrids

ashps_spf = homes_within_threshold[(homes_within_threshold["HP_Type_2"] == "ASHPs")]["SPF_value"]
hyrbid_spf = homes_within_threshold[(homes_within_threshold["HP_Type_2"] == "Hybrid")]["SPF_value"]

statistic, pvalue = stats.ttest_ind(ashps_spf, hyrbid_spf)
if pvalue < 0.05:
    print("Significant difference in means between ASHPs and hybrids across all SPF types")
else:
    print("No significant difference in means between ASHPs and hybrids across all SPF types")


# In[ ]:


# Generate stats on All_ASHPs (ASHPs + Hybrids) that qualify for the threshold
af.stats_by_hp_and_SPF_type(homes_within_threshold, by="HP_Type_3")[
    ["HP_Type_3", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]
].set_index(["HP_Type_3", "SPF_type"])


# The scatter chart below shows the relationship between SPFH2 and total energy output. It also shows the SPF thresholds used and which homes are removed from the analysis (as per the waterfall chart).

# In[ ]:


# Generate scatter plot of SPFH2 values vs total energy output
cf.SPF_h2_threshold_scatter(
    home_summary,
    SPF_value_min_threshold,
    SPF_value_max_threshold,
    scatter_opacity=scatter_opacity,
)


# The chart below shows the frequency distribution of binned SPF for each SPF type and heat pump type in HP_Type_2.
# 
# <!-- Optional markdown -->

# In[ ]:


# # Optional code to create frequency distribution of binned SPF for each SPF type and heat pump type in HP_Type_2

# # Bin SPF
# fig_data_binned = homes_within_threshold.copy()
# fig_data_binned["count"] = 1
# fig_data_binned["SPF_value_binned"] = pd.cut(fig_data_binned["SPF_value"], 10, precision=1)
# fig_data_binned = fig_data_binned.sort_values("SPF_value_binned", ascending=False)
# fig_data_binned["SPF_value_binned"] = fig_data_binned["SPF_value_binned"].astype(str)

# # Normalise y axis
# fig_histogram_data = pd.DataFrame(
#     fig_data_binned.groupby(["HP_Type_2", "SPF_type", "SPF_value_binned"]).count()
# ).reset_index()
# fig_histogram_data["count"] = fig_histogram_data["count"] / fig_histogram_data.groupby(["HP_Type_2", "SPF_type"])[
#     "count"
# ].transform("sum")

# # Create histogram
# fig_histogram = px.histogram(
#     fig_histogram_data.sort_values(["SPF_type", "SPF_value_binned"]),
#     x="SPF_value_binned",
#     y="count",
#     color="HP_Type_2",
#     facet_col="SPF_type",
#     facet_row="HP_Type_2",
#     width=1200,
#     height=800,
#     title="Frequency distribution by binned SPF and heat pump type",
#     labels={"SPF_value_binned": "Binned SPF"},
# )
# fig_histogram.update_yaxes(title_text="Percentage")

# fig_histogram.update_yaxes(title_text="Proportion")
# fig_histogram.show()


# In[ ]:


# # Optional code to create frequency distribution of binned SPF for each SPF type and heat pump type in HP_Type_2

# # Bin SPF
# fig_data_binned = homes_within_threshold.copy()
# fig_data_binned["count"] = 1
# fig_data_binned["SPF_value_binned"] = pd.cut(fig_data_binned["SPF_value"], 10, precision=1)
# fig_data_binned = fig_data_binned.sort_values("SPF_value_binned", ascending=False)
# fig_data_binned["SPF_value_binned"] = fig_data_binned["SPF_value_binned"].astype(str)

# # Normalise y axis
# fig_histogram_data = pd.DataFrame(
#     fig_data_binned.groupby(["HP_Type_3", "SPF_type", "SPF_value_binned"]).count()
# ).reset_index()
# fig_histogram_data["count"] = fig_histogram_data["count"] / fig_histogram_data.groupby(["HP_Type_3", "SPF_type"])[
#     "count"
# ].transform("sum")

# # Create histogram
# fig_histogram = px.histogram(
#     fig_histogram_data.sort_values(["SPF_type", "SPF_value_binned"]),
#     x="SPF_value_binned",
#     y="count",
#     color="HP_Type_3",
#     facet_col="SPF_type",
#     width=1200,
#     title="Frequency distribution",
#     labels={"SPF_value_binned": "Binned SPF"},
# )
# fig_histogram.update_yaxes(title_text="Percentage")
# fig_histogram.update_layout(yaxis_tickformat="0%", showlegend=False)
# fig_histogram.show()


# The chart below shows the empirical distribution for SPF for each heat pump type in HP_Type.
# 
# <!-- Optional markdown -->

# In[ ]:


# Optional code to generate empirical cumulative distribution function for SPF by heat pump type
px.ecdf(
    homes_within_threshold,
    x="SPF_value",
    color="HP_Type",
    height=400,
    width=1000,
    facet_col="SPF_type",
    title=f"Empirical Distribution",
    labels={
        "SPF_value": "SPF",
    },
)


# In[ ]:


# Optional code to generate empirical cumulative distribution function for SPF by heat pump type
fig = px.ecdf(
    homes_within_threshold,
    x="SPF_value",
    color="SPF_type",
    height=500,
    width=500,
    title=f"Empirical Distribution",
    labels={"SPF_value": "SPF"},
)
fig.update_layout(yaxis_tickformat="0%")
fig.update_yaxes(title="Percentage")
fig.show()


# The chart below shows the empirical distribution for SPF for each heat pump type in HP_Type_2.
# 
# The empirical distribution cumulative function (ecdf) charts below shows that Hybrid heat pumps have lower a SPF and a much wider spread of performance vs ASHPs.

# In[ ]:


# Generate empirical cumulative distribution function for SPF by heat pump type

px.ecdf(
    homes_within_threshold[
        ~(homes_within_threshold["HP_Type"] == "Hybrid") | ~(homes_within_threshold["SPF_type"] == "SPFH3")
    ],
    x="SPF_value",
    color="HP_Type_2",
    height=400,
    width=1000,
    facet_col="SPF_type",
    title=f"Empirical Distribution",
    labels={"SPF_value": "SPF"},
)


# In[ ]:


# Generate empirical cumulative distribution function for SPF by heat pump type
px.ecdf(
    homes_within_threshold[(homes_within_threshold["HP_Type"] != "Hybrid")],
    x="SPF_value",
    color="HP_Type",
    height=400,
    width=1000,
    facet_col="SPF_type",
    title=f"Empirical Distribution",
    labels={"SPF_value": "SPF"},
)


# #### Analysis by House Type and Demographic
# 
# The analysis of variations in performance below is based on SPFH4, given that this provides the best measure of overall heat pump system performance (and is least sensitive to data quality issues). This analysis has been repeated for SPFH2 but is not shown in full here (for brevity) – any significant differences between the conclusions are highlighted in the commentary below.
# 
# The analysis is also focused on ASHPs, as they are the only heat pump type with enough data to further segment.
# 
# ##### Does House Type affect SPFH4?

# The impact of house type is not entirely clear, however we can say with moderate confidence that detached homes have a statistically significantly higher SPF compared to semi-detached homes.

# In[ ]:


# Filter for ASHPs
# The analysis by sub-group (house type, income, etc.) is focused on ASHPs, as they are the only heat pump type with enough data to further segment.
ashps_within_threshold = homes_within_threshold[homes_within_threshold["HP_Type_2"] == "ASHPs"]

# Filter for SPFH4 only, as it represents the broadest system boundary
ashps_within_threshold = ashps_within_threshold[ashps_within_threshold["SPF_type"] == "SPFH4"]


# In[ ]:


# Generate bar graph showing mean heat energy output by house type

homes_within_threshold_grouped = (
    homes_within_threshold.groupby("House_Form")["window_Total_Energy_Output"].mean().sort_values() / 365
)

# Re-order in logical order
homes_within_threshold_grouped = pd.concat(
    [
        homes_within_threshold_grouped[homes_within_threshold_grouped.index == "Detached"],
        homes_within_threshold_grouped[homes_within_threshold_grouped.index == "Semi-Detached"],
        homes_within_threshold_grouped[homes_within_threshold_grouped.index == "End-Terrace"],
        homes_within_threshold_grouped[homes_within_threshold_grouped.index == "Mid-Terrace"],
        homes_within_threshold_grouped[homes_within_threshold_grouped.index == "Flat"],
    ]
)

px.bar(
    homes_within_threshold_grouped,
    y="window_Total_Energy_Output",
    width=600,
    labels={"window_Total_Energy_Output": "Mean Heat Energy Output (kWh/day)", "House_Form": "House Type"},
) 


# The table below shows SPF statistics for each house form and heat pump type in HP_Type_2.

# In[ ]:


# Create a blank DataFrame for the loop
stats_by_house_type = []

# Calculate statistics for each house type
for house_type in ashps_within_threshold["House_Form"].unique():
    home_summary_split = ashps_within_threshold[ashps_within_threshold["House_Form"] == house_type]
    df = af.stats_by_hp_and_SPF_type(home_summary_split, by="HP_Type_2")
    df["House_Type"] = house_type
    df = df.set_index("House_Type")
    stats_by_house_type.append(df)
stats_by_house_type = pd.concat(stats_by_house_type)

# Reorder DataFrame to logical order
stats_by_house_type = pd.concat(
    [
        stats_by_house_type[stats_by_house_type.index == "Detached"],
        stats_by_house_type[stats_by_house_type.index == "Semi-Detached"],
        stats_by_house_type[stats_by_house_type.index == "End-Terrace"],
        stats_by_house_type[stats_by_house_type.index == "Mid-Terrace"],
        stats_by_house_type[stats_by_house_type.index == "Flat"],
    ]
).sort_values(["HP_Type_2"])

# View outputs
stats_by_house_type[["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index().set_index(
    ["HP_Type_2", "House_Type", "SPF_type"]
)


# In[ ]:


# Calculate confidence interval
stats_by_house_type["half_interval"] = stats_by_house_type["Mean"] - stats_by_house_type["Mean CI Lower"]

# Manually set y-axis range
y_range_min = 1.8
y_range_max = 3.8

# Generate bar graph of means and confidence intervals for each group
fig = px.bar(
    stats_by_house_type,
    x=stats_by_house_type.index,
    y="Mean",
    width=600,
    range_y=[y_range_min, y_range_max],
    error_y=stats_by_house_type["half_interval"],
    title="ASHPs Mean SPFH4 with 95% Confidence Intervals",
    labels={"Mean": "Mean SPFH4", "House_Type": "House Type"},
    hover_data=["Homes"],
)

# Add n= to graph
fig.add_trace(
    go.Line(
        x=stats_by_house_type.index,
        y=[y_range_min] * stats_by_house_type.index.size,
        mode="markers+text",
        text="n=" + stats_by_house_type["Homes"].map(str),
        textposition="top center",
        textfont_color="white",
        marker_opacity=0,
        textfont_size=10,
    )
)

fig.update_layout(showlegend=False)


# In[ ]:


px.scatter(
    ashps_within_threshold[
        (ashps_within_threshold["House_Form"] == "Detached") | (ashps_within_threshold["House_Form"] == "Mid-Terrace")
    ],
    x="Total_Floor_Area",
    y="SPF_value",
    width=800,
    trendline="ols",
)


# Groups are normally distributed and have equal variances.

# In[ ]:


# Begin statistical testing for difference in means over house types

# Filter DataFrame for necessary columns
house_form_data = ashps_within_threshold[["House_Form", "SPF_value"]]

# Create a DataFrame for each group
data_detached = house_form_data[house_form_data["House_Form"] == "Detached"]["SPF_value"]
data_semi = house_form_data[house_form_data["House_Form"] == "Semi-Detached"]["SPF_value"]
data_end = house_form_data[house_form_data["House_Form"] == "End-Terrace"]["SPF_value"]
data_mid = house_form_data[house_form_data["House_Form"] == "Mid-Terrace"]["SPF_value"]
data_flat = house_form_data[house_form_data["House_Form"] == "Flat"]["SPF_value"]

# Shapiro-Wilk non-parametric test for normality
# h0 = population is normally distributed
for house_form in house_form_data["House_Form"].unique():
    print(house_form)
    statistic, p_value = stats.shapiro(house_form_data[house_form_data["House_Form"] == house_form]["SPF_value"])
    if p_value < 0.05:
        print("Shapiro-Wilk: data is not normally distributed")
    else:
        print("Shapiro-Wilk: data is normally distributed")

# Levene's non-parametric test for homogeneity / homoscedasticity of variances
# h0 = population has equal variances
stat, p_value = stats.levene(
    data_detached,
    data_semi,
    data_end,
    data_mid,
    data_flat,
    center="mean",
    proportiontocut=0.05,
)
if p_value < 0.05:
    print("Levene's test: data has unequal variances")
else:
    print("Levene's test: data has equal variances")

data_qq_plot = stats.probplot(house_form_data["SPF_value"], dist="norm", plot=pylab)


# We can therefore conduct the ANOVA test. The test results indicate there are groups with significantly different means.

# In[ ]:


# ANOVA parametric test to compare the means of 2 or more groups
# h0 = there are no groups with significantly different means
# Assumptions: normality within groups, equal variances
statistic, p_value = stats.f_oneway(data_detached, data_semi, data_end, data_mid, data_flat)
if p_value < 0.05:
    print("ANOVA test: There are groups with significantly difference means")
else:
    print("ANOVA test: There are no groups with significantly different means")


# We check which groups have significantly difference means using Tukey's HSD test.

# In[ ]:


# Tukey's HSD parametric test to test equality of means over multiple groups
# h0 = there is no significant different in means over the paired group
# Assumptions: normality within groups
Results = pairwise_tukeyhsd(endog=house_form_data["SPF_value"], groups=house_form_data["House_Form"], alpha=0.05)
print(Results)


# The following paired groups have significantly different means.

# In[ ]:


Significant_results = pd.DataFrame(data=Results._results_table.data[1:], columns=Results._results_table.data[0])
Significant_results[Significant_results["p-adj"] < 0.05][["group1", "group2"]].reset_index(drop=True)


# In[ ]:


ashps_within_threshold = pd.concat(
    [
        ashps_within_threshold[ashps_within_threshold["House_Form"] == "Detached"],
        ashps_within_threshold[ashps_within_threshold["House_Form"] == "Semi-Detached"],
        ashps_within_threshold[ashps_within_threshold["House_Form"] == "End-Terrace"],
        ashps_within_threshold[ashps_within_threshold["House_Form"] == "Mid-Terrace"],
        ashps_within_threshold[ashps_within_threshold["House_Form"] == "Flat"],
    ]
)

px.histogram(
    ashps_within_threshold,
    x="House_Form",
    color="HP_Type",
    width=700,
    labels={"House_Form": "House Type"},
    barnorm="percent",
)


# In[ ]:


px.histogram(
    ashps_within_threshold,
    x="House_Form",
    color="HP_Refrigerant",
    width=700,
    labels={"House_Form": "House Type"},
    barnorm="percent",
)


# ##### Does Household Income affect SPFH4?

# We can say with moderate confidence that household income does not have a significant impact on the mean SPF value amongst those who provided details of their income.
# 
# The table below shows SPF statistics for each house form and heat pump type.

# In[ ]:


# Create a blank DataFrame for the loop
stats_by_income_group = []

# Calculate statistics for each income group
for income_group in ashps_within_threshold["House_Income"].unique():
    home_summary_split = ashps_within_threshold[ashps_within_threshold["House_Income"] == income_group]
    df = af.stats_by_hp_and_SPF_type(home_summary_split, by="HP_Type_2")
    df["House_Income"] = income_group
    df = df.set_index("House_Income")
    stats_by_income_group.append(df)

# Reorder DataFrame to logical order
stats_by_income_group = pd.concat(stats_by_income_group).sort_values(["HP_Type_2", "House_Income"])

# View outputs
stats_by_income_group[["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index().set_index(
    ["House_Income", "HP_Type_2", "SPF_type"]
)


# In[ ]:


# Calculate confidence interval
stats_by_income_group["half_interval"] = (
    stats_by_income_group["Mean"] - stats_by_income_group["Mean CI Lower"]
).sort_values()

y_range_min = 1.8
y_range_max = 4

# Generate bar graph of means and confidence intervals for each group
fig = px.bar(
    stats_by_income_group,
    x=stats_by_income_group.index,
    y="Mean",
    width=600,
    range_y=[y_range_min, y_range_max],
    error_y=stats_by_income_group["half_interval"],
    title="ASHPs Mean SPFH4 with 95% Confidence Intervals",
    labels={"Mean": "Mean SPFH4", "House_Income": "Household Income (£)"},
    hover_data=["Homes"],
)

fig.add_trace(
    go.Line(
        x=stats_by_income_group.index,
        y=[y_range_min] * stats_by_income_group.index.size,
        mode="markers+text",
        text="n=" + stats_by_income_group["Homes"].map(str),
        textposition="top center",
        textfont_color="white",
        textfont_size=10,
        marker_opacity=0,
    )
)
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


# Begin statistical testing for difference in means over household incomes

# Filter DataFrame for necessary columns
household_income_data = ashps_within_threshold[["House_Income", "SPF_value"]]

# Create a DataFrame for each group
data_income_1 = household_income_data[household_income_data["House_Income"] == "0 - 12,500"]["SPF_value"]
data_income_2 = household_income_data[household_income_data["House_Income"] == "12,501 - 16,200"]["SPF_value"]
data_income_3 = household_income_data[household_income_data["House_Income"] == "16,201 - 20,000"]["SPF_value"]
data_income_4 = household_income_data[household_income_data["House_Income"] == "20,001 - 25,000"]["SPF_value"]
data_income_5 = household_income_data[household_income_data["House_Income"] == "25,001 - 30,000"]["SPF_value"]
data_income_6 = household_income_data[household_income_data["House_Income"] == "30,001 - 40,000"]["SPF_value"]
data_income_7 = household_income_data[household_income_data["House_Income"] == "40,001 - 50,000"]["SPF_value"]
data_income_8 = household_income_data[household_income_data["House_Income"] == "50,001+"]["SPF_value"]
data_income_9 = household_income_data[household_income_data["House_Income"] == "Prefer not to say"]["SPF_value"]

# Shapiro-Wilk non-parametric test for normality
# h0 = population is normally distributed
for house_income_group in ashps_within_threshold["House_Income"].sort_values().unique():
    print(house_income_group)
    test_data = household_income_data[household_income_data["House_Income"] == house_income_group]
    statistic, p_value = stats.shapiro(test_data["SPF_value"])
    if p_value < 0.05:
        print("Shapiro-Wilk test: data is not normally distributed")
    else:
        print("Shapiro-Wilk test: data is normally distributed")

# Levene's non-parametric test for homogeneity / homoscedasticity of variances
# h0 = population has equal variances
stat, p_value = stats.levene(
    data_income_1,
    data_income_2,
    data_income_3,
    data_income_4,
    data_income_5,
    data_income_6,
    data_income_7,
    data_income_8,
    data_income_9,
    center="mean",
    proportiontocut=0.05,
)
if p_value < 0.05:
    print("Levene's test: data has unequal variances")
else:
    print("Levene's test: data has equal variances")

data_qq_plot = stats.probplot(household_income_data["SPF_value"], dist="norm", plot=pylab)


# We can therefore conduct the ANOVA test. The test results indicate there are no groups with significantly different means.

# In[ ]:


# ANOVA parametric test to compare the emans of 2 or more groups
# h0 = there are no groups with significantly different means
# Assumptions: normality within groups, equal variances
statistic, p_value = stats.f_oneway(
    data_income_1,
    data_income_2,
    data_income_3,
    data_income_4,
    data_income_5,
    data_income_6,
    data_income_7,
    data_income_8,
    data_income_9,
)
if p_value < 0.05:
    print("ANOVA test: There are groups with significantly different in means")
else:
    print("ANOVA test: There are no groups with significantly different means")


# ##### Does House Age affect SPFH4?

# - We can say with moderate confidence that house age does not have a significant impact on the mean SPF value.
# 
# The table below shows SPF statistics for each house age and heat pump type in HP_Type_2.

# In[ ]:


# Create a blank DataFrame for the loop
stats_by_house_age = []

# Calculate statistics for each house age group
for house_age in ashps_within_threshold["House_Age_Cleaned"].unique():
    home_summary_split = ashps_within_threshold[ashps_within_threshold["House_Age_Cleaned"] == house_age]
    df = af.stats_by_hp_and_SPF_type(home_summary_split, by="HP_Type_2")
    df["House_Age_Cleaned"] = house_age
    df = df.set_index("House_Age_Cleaned")
    stats_by_house_age.append(df)

# Reorder DataFrame to logical order
stats_by_house_age = pd.concat(stats_by_house_age)
stats_by_house_age = pd.concat(
    [
        stats_by_house_age[stats_by_house_age.index == "Pre-1919"],
        stats_by_house_age[stats_by_house_age.index == "1919-1944"],
        stats_by_house_age[stats_by_house_age.index == "1945-1964"],
        stats_by_house_age[stats_by_house_age.index == "1965-1980"],
        stats_by_house_age[stats_by_house_age.index == "1981-1990"],
        stats_by_house_age[stats_by_house_age.index == "1991-2000"],
        stats_by_house_age[stats_by_house_age.index == "2001+"],
    ]
)

# View outputs
stats_by_house_age[["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index().rename(
    {"House_Age_Cleaned": "House_Age"}, axis=1
).set_index(["House_Age", "HP_Type_2", "SPF_type"])


# In[ ]:


# Calculate confidence interval
stats_by_house_age["half_interval"] = stats_by_house_age["Mean"] - stats_by_house_age["Mean CI Lower"]

y_range_min = 2
y_range_max = 3.2

# Generate bar graph of means and confidence intervals for each group
fig = px.bar(
    stats_by_house_age,
    x=stats_by_house_age.index,
    y="Mean",
    width=600,
    range_y=[y_range_min, y_range_max],
    error_y=stats_by_house_age["half_interval"],
    title="ASHPs Mean SPFH4 with 95% Confidence Intervals",
    labels={"Mean": "Mean SPFH4", "House_Age_Cleaned": "House Age"},
    hover_data=["Homes"],
)

fig.add_trace(
    go.Line(
        x=stats_by_house_age.index,
        y=[y_range_min] * stats_by_house_age.index.size,
        mode="markers+text",
        text="n=" + stats_by_house_age["Homes"].map(str),
        textposition="top center",
        textfont_color="white",
        marker_opacity=0,
        textfont_size=10,
    )
)
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


# Begin statistical testing for difference in means over household incomes

# Filter DataFrame for necessary columns
house_age_data = ashps_within_threshold[["House_Age_Cleaned", "SPF_value"]]

# Create a DataFrame for each group
data_age_1 = house_age_data[house_age_data["House_Age_Cleaned"] == "1919-1944"]["SPF_value"]
data_age_2 = house_age_data[house_age_data["House_Age_Cleaned"] == "1945-1964"]["SPF_value"]
data_age_3 = house_age_data[house_age_data["House_Age_Cleaned"] == "1965-1980"]["SPF_value"]
data_age_4 = house_age_data[house_age_data["House_Age_Cleaned"] == "1981-1990"]["SPF_value"]
data_age_5 = house_age_data[house_age_data["House_Age_Cleaned"] == "1991-2000"]["SPF_value"]
data_age_6 = house_age_data[house_age_data["House_Age_Cleaned"] == "2001+"]["SPF_value"]
data_age_7 = house_age_data[house_age_data["House_Age_Cleaned"] == "Pre-1919"]["SPF_value"]

# Shapiro-Wilk non-parametric test for normality
# h0 = population is normally distributed
for house_age_group in ashps_within_threshold["House_Age_Cleaned"].sort_values().unique():
    print(house_age_group)
    test_data = house_age_data[house_age_data["House_Age_Cleaned"] == house_age_group]
    statistic, p_value = stats.shapiro(test_data["SPF_value"])
    if p_value < 0.05:
        print("Shapiro-Wilk test: data is not normally distributed")
    else:
        print("Shapiro-Wilk test: data is normally distributed")

# Levene's non-parametric test for homogeneity / homoscedasticity of variances
# h0 = population has equal variances
stat, p_value = stats.levene(
    data_age_1,
    data_age_2,
    data_age_3,
    data_age_4,
    data_age_5,
    data_age_6,
    data_age_7,
    center="mean",
    proportiontocut=0.05,
)
if p_value < 0.05:
    print("Levene's test: data has unequal variances")
else:
    print("Levene's test: data has equal variances")

data_qq_plot = stats.probplot(house_age_data["SPF_value"], dist="norm", plot=pylab)


# We can therefore conduct the ANOVA test. The test results indicate there are no groups with significantly different means.

# In[ ]:


# ANOVA parametric test to compare the emans of 2 or more groups
# h0 = there are no groups with significantly different means
# Assumptions: normality within groups, equal variances
statistic, p_value = stats.f_oneway(data_age_1, data_age_2, data_age_3, data_age_4, data_age_5, data_age_6, data_age_7)
if p_value < 0.05:
    print("ANOVA test: There are groups with significantly different in means")
else:
    print("ANOVA test: There are no groups with significantly different means")


# ##### Does socioeconomic group affect SPFH4?

# In[ ]:


# Create a blank DataFrame for the loop
stats_by_socio_group = []

# Calculate statistics for each income group
for income_group in ashps_within_threshold["Social_Group"].unique():
    home_summary_split = ashps_within_threshold[ashps_within_threshold["Social_Group"] == income_group]
    df = af.stats_by_hp_and_SPF_type(home_summary_split, by="HP_Type_2")
    df["Social_Group"] = income_group
    df = df.set_index("Social_Group")
    stats_by_socio_group.append(df)

# Reorder DataFrame to logical order
stats_by_socio_group = pd.concat(stats_by_socio_group).sort_values(["HP_Type_2", "Social_Group"])

# View outputs
stats_by_socio_group[["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index().set_index(
    ["Social_Group", "HP_Type_2", "SPF_type"]
)


# In[ ]:


# Calculate confidence interval
stats_by_socio_group["half_interval"] = (
    stats_by_socio_group["Mean"] - stats_by_socio_group["Mean CI Lower"]
).sort_values()

y_range_min = 2.5
y_range_max = 3

# Generate bar graph of means and confidence intervals for each group
fig = px.bar(
    stats_by_socio_group,
    x=stats_by_socio_group.index,
    y="Mean",
    width=600,
    range_y=[y_range_min, y_range_max],
    error_y=stats_by_socio_group["half_interval"],
    title="ASHPs Mean SPFH4 with 95% Confidence Intervals",
    labels={"Mean": "Mean SPFH4", "Social_Group": "Socio-economic Group"},
    hover_data=["Homes"],
)

fig.add_trace(
    go.Line(
        x=stats_by_socio_group.index,
        y=[y_range_min] * stats_by_socio_group.index.size,
        mode="markers+text",
        text="n=" + stats_by_socio_group["Homes"].map(str),
        textposition="top center",
        textfont_color="white",
        textfont_size=10,
        marker_opacity=0,
    )
)
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


# Begin statistical testing for difference in means over household incomes

# Filter DataFrame for necessary columns
house_socio_data = ashps_within_threshold[["Social_Group", "SPF_value"]]

# Create a DataFrame for each group
data_socio_1 = house_socio_data[house_socio_data["Social_Group"] == "AB"]["SPF_value"]
data_socio_2 = house_socio_data[house_socio_data["Social_Group"] == "C1"]["SPF_value"]
data_socio_3 = house_socio_data[house_socio_data["Social_Group"] == "C2"]["SPF_value"]
data_socio_4 = house_socio_data[house_socio_data["Social_Group"] == "DE"]["SPF_value"]

# Shapiro-Wilk non-parametric test for normality
# h0 = population is normally distributed
for house_age_group in ashps_within_threshold["Social_Group"].sort_values().unique():
    print(house_age_group)
    test_data = house_socio_data[house_socio_data["Social_Group"] == house_age_group]
    statistic, p_value = stats.shapiro(test_data["SPF_value"])
    if p_value < 0.05:
        print("Shapiro-Wilk test: data is not normally distributed")
    else:
        print("Shapiro-Wilk test: data is normally distributed")

# Levene's non-parametric test for homogeneity / homoscedasticity of variances
# h0 = population has equal variances
stat, p_value = stats.levene(
    data_socio_1,
    data_socio_2,
    data_socio_3,
    data_socio_4,
    center="mean",
    proportiontocut=0.05,
)
if p_value < 0.05:
    print("Levene's test: data has unequal variances")
else:
    print("Levene's test: data has equal variances")

data_qq_plot = stats.probplot(house_socio_data["SPF_value"], dist="norm", plot=pylab)


# In[ ]:


# ANOVA parametric test to compare the emans of 2 or more groups
# h0 = there are no groups with significantly different means
# Assumptions: normality within groups, equal variances
statistic, p_value = stats.f_oneway(data_socio_1, data_socio_2, data_socio_3, data_socio_4)
if p_value < 0.05:
    print("ANOVA test: There are groups with significantly different in means")
else:
    print("ANOVA test: There are no groups with significantly different means")


# ##### Do HT_ASHPs perform differently to LT_ASHPs?

# HT_ASHPs spend much of their time operating at similar temperatures as LT_ASHPs. HT_ASHPs are not observed to perform worse than LT_ASHPs.
# 
# We choose to focus on mean flow temperature, rather than median or mode, as this best accounts for both single-modal and bi-model frequency distributions.

# In[ ]:


# Remove hybrid heat pumps
flow_temp_data = homes_within_threshold[homes_within_threshold["HP_Type_2"] != "Hybrid"]


# The vast majority of HT_ASHPs use a different refrigerant versus LT_ASHPs, whose performance differs greatly to the refrigerants used in LT_ASHPs. The lower performance of LT_ASHPs can therefore be attributed to this this rather than the differences in the temperatures they're operating at.
# 
# The table below shows the SPF of each refrigerant across ASHPs.

# In[ ]:


stats_by_refrigerant = af.stats_by_hp_and_SPF_type(flow_temp_data, by="HP_Refrigerant").set_index(["HP_Refrigerant"])
stats_by_refrigerant[stats_by_refrigerant["SPF_type"] == "SPFH4"].sort_index(level=["R290", "R32", "R410a"])[
    ["SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]
]


# The table below shows the the count of refrigerant across HT and LT ASHPs.

# In[ ]:


round(
    flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"]
    .groupby(["HP_Type", "HP_Refrigerant"])["SPF_value"]
    .agg(["median", "count"])
    .rename({"median": "Median SPFH4", "count": "count"}, axis=1),
    2,
)


# In[ ]:


dfs = []

df = flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"]
for hp_type in ["HT_ASHP", "LT_ASHP"]:
    df = flow_temp_data[flow_temp_data["HP_Type"] == hp_type]

    df = af.stats_by_hp_and_SPF_type(df, by="HP_Refrigerant").set_index(["HP_Refrigerant"])
    df["HP_Type"] = hp_type
    df = df[df["SPF_type"] == "SPFH4"].sort_index(level=["R290", "R32", "R410a"])[
        ["HP_Type", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]
    ]
    dfs.append(df)
pd.concat(dfs)


# The box plot below shows the R410A refrigerant performing worse than R290 and R32.

# In[ ]:


px.box(
    flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"],
    x="HP_Refrigerant",
    y="SPF_value",
    width=500,
    title="ASHPs",
    labels={"SPF_value": "SPFH4"},
)


# In[ ]:


# print(len(flow_temp_data))
# flow_temp_data = flow_temp_data[flow_temp_data["temperature_cleaning_issue"].isna()]
# print(len(flow_temp_data))


# The frequency distribution below shows the count across the binned mean flow temperature, split between LT and HT ASHPs.

# In[ ]:


# Bin mean flow temperature
flow_temp_data["Binned Mean Flow Temperature"] = pd.cut(
    flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    bins=10,
    precision=0,
)

# Normalise y axis
flow_temp_data_binned_grouped = (
    flow_temp_data.groupby(["Binned Mean Flow Temperature", "HP_Type"]).count()["Property_ID"].reset_index()
)
flow_temp_data_binned_grouped = flow_temp_data_binned_grouped.merge(
    flow_temp_data_binned_grouped.groupby("HP_Type").apply(sum)["Property_ID"],
    on="HP_Type",
)
flow_temp_data_binned_grouped["Proportion"] = (
    flow_temp_data_binned_grouped["Property_ID_x"] / flow_temp_data_binned_grouped["Property_ID_y"]
)
flow_temp_data_binned_grouped = flow_temp_data_binned_grouped.sort_values(
    ["Binned Mean Flow Temperature"], ascending=True
)

# Reorder DataFrame
flow_temp_data_binned_grouped = pd.concat(
    [
        flow_temp_data_binned_grouped[flow_temp_data_binned_grouped["HP_Type"] == "LT_ASHP"],
        flow_temp_data_binned_grouped[flow_temp_data_binned_grouped["HP_Type"] == "HT_ASHP"],
    ]
)

# Generate frequency distribution (as line plot)
fig = px.line(
    flow_temp_data_binned_grouped,
    x=flow_temp_data_binned_grouped["Binned Mean Flow Temperature"].astype(str),
    y="Proportion",
    color="HP_Type",
    width=700,
    labels={"x": "Binned Mean Flow Temperature", "Proportion": "Percentage"},
    title="Frequency distribution of binned mean flow temp",
)

fig = fig.update_layout(yaxis_tickformat="0%")
fig.show()


# The chart below shows the performance of different refrigerants across the range of mean flow temperatures. It shows again that the units fitted with R410A refrigerant perform significantly worse than R290 and R32.

# In[ ]:


# Generate scatter plot with regression lines by refrigerant and heat pump type
fig = go.Figure()

colours = [
    cf.colour1_rgb,
    cf.colour2_rgb,
    cf.colour3_rgb,
]
symbols = ["circle", "x"]

for refrigerant, colour, offset in zip(flow_temp_data["HP_Refrigerant"].unique(), colours, [-0.01, 0, 0.01]):
    print(refrigerant)
    for hp_type, symbol in zip(["LT_ASHP", "HT_ASHP"], symbols):
        print(hp_type)
        flow_temp_data_loop = flow_temp_data[
            (flow_temp_data["HP_Refrigerant"] == refrigerant)
            & (flow_temp_data["HP_Type"] == hp_type)
            & (flow_temp_data["SPF_type"] == "SPFH4")
        ]
        try:
            fig.add_trace(
                go.Scatter(
                    x=flow_temp_data_loop["window_mean_Heat_Pump_Heating_Flow_Temperature"],
                    y=flow_temp_data_loop["SPF_value"],
                    mode="markers",
                    name=f"{refrigerant}_{hp_type}",
                    marker_color=f"rgb{colour}",
                    marker_symbol=symbol,
                )
            )
            # We are not interested in plotting a regression line for R32 HT_ASHPs as there are not many.
            if refrigerant == "R32" and hp_type == "HT_ASHP":
                continue
            else:
                regr = LinearRegression()
                res = regr.fit(
                    np.array(flow_temp_data_loop["window_mean_Heat_Pump_Heating_Flow_Temperature"]).reshape(-1, 1),
                    flow_temp_data_loop["SPF_value"],
                )

                fit = regr.predict(
                    np.array(flow_temp_data_loop["window_mean_Heat_Pump_Heating_Flow_Temperature"]).reshape(-1, 1)
                )

                fig.add_trace(
                    go.Scatter(
                        x=flow_temp_data_loop["window_mean_Heat_Pump_Heating_Flow_Temperature"],
                        y=fit,
                        mode="lines",
                        name=f"{refrigerant}_{hp_type}_fit",
                        marker_color=f"rgb{colour}",
                    )
                )

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    flow_temp_data_loop["window_mean_Heat_Pump_Heating_Flow_Temperature"],
                    flow_temp_data_loop["SPF_value"],
                )

                print(f"p_value (2 d.p.) : {round(p_value, 2)}")
                x = flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"].max()
                y = slope * x + intercept

                fig.add_trace(
                    go.Scatter(
                        x=[x * 1.05],
                        y=[y * (1 + 7 * offset)],
                        mode="lines+text",
                        text=f"R²={round(r_value**2,2)}",
                        textposition="middle center",
                        textfont=dict(color=f"rgb{colour}"),
                        showlegend=False,
                    )
                )

        except:
            pass

fig.update_layout(
    width=800,
    title="Relationship between mean flow temperature and SPFH4 by Refrigerant",
)
fig.update_xaxes(title="Mean Heat Pump Flow Temperature")
fig.update_yaxes(title="SPFH4")
fig.update_traces(opacity=scatter_opacity)
fig.show()


# In[ ]:


# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"]["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"]["SPF_value"],
)

print(
    round(
        pd.DataFrame(
            {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
            },
            index=[0],
        ).T.rename({0: "value"}, axis=1),
        2,
    )
)

x = flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"].max()
y = slope * x + intercept

fig = px.scatter(
    flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"],
    x="window_mean_Heat_Pump_Heating_Flow_Temperature",
    y="SPF_value",
    width=800,
    trendline="ols",
)

fig.update_yaxes(title="SPFH4")
fig.update_xaxes(title="Mean Heat Pump Flow Temperature (°C)")
fig.add_trace(
    go.Scatter(
        x=[x * 1.04],
        y=[y],
        mode="lines+text",
        text=f"R²={round(r_value**2,2)}",
        textposition="middle center",
        textfont=dict(color=f"rgb{list(cf.colour_scheme().values())[0]}"),
    )
)
fig.update_layout(showlegend=False)


# In[ ]:


flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"].describe()


# The chart below shows that on average, HT_ASHPs tend to operate at similar temperatures to LT_ASHPs.
# 
# In the chart below, there is a significant difference in SPFH4 between LT_ASHPs and HT_ASHPs, especially in the flow temperature range [33, 40]. This could be due HT_ASHPs being more capable of dealing with differing external air temperatures, allowing the heat pump to operate at a lower temperature more often.
# 
# While HT_ASHPs operate at slightly higher flow temperatures, many of them do end up operating at similar temperatures as LT_ASHPs.
# 
# There is a statistically significant relationship (p<0.05) between mean heat pump flow temperature of HT_ASHPs and spf_value – systems where the heat pump flow temperature is higher tend to have higher SPFs. Note that there is a lot of variation in heat pump performance that is not explained by this factor in isolation.

# In[ ]:


# Generate scatter plot to show relationship between mean flow temperature and SPFH4
flow_scatter = px.scatter(
    flow_temp_data,
    x="window_mean_Heat_Pump_Heating_Flow_Temperature",
    y="SPF_value",
    color="HP_Type",
    width=700,
    trendline="ols",
    title="Relationship between mean flow temperature and SPFH4 by heat pump type",
    labels={
        "SPF_value": "SPFH4",
        "window_mean_Heat_Pump_Heating_Flow_Temperature": "Mean Heat Pump Flow Temperature",
    },
)

flow_scatter.update_traces(opacity=scatter_opacity)
flow_scatter.show()


# The value lies below 0.05, showing the relationship to be statistically significant.

# In[ ]:


# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    flow_temp_data[flow_temp_data["HP_Type"] == "HT_ASHP"]["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    flow_temp_data[flow_temp_data["HP_Type"] == "HT_ASHP"]["SPF_value"],
)

round(
    pd.DataFrame(
        {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
        },
        index=[0],
    ).T.rename({0: "value"}, axis=1),
    2,
)


# The chart below shows the same relationship but groups LT and HT ASHPs together.

# In[ ]:


# Generate scatter plot to show relationship between mean flow temperature and SPFH4
flow_scatter_2 = px.scatter(
    flow_temp_data,
    x="window_mean_Heat_Pump_Heating_Flow_Temperature",
    y="SPF_value",
    color="HP_Type_2",
    width=700,
    trendline="ols",
    title="ASHPs: Relationship between mean flow temperature and SPFH4",
    labels={
        "SPF_value": "SPFH4",
        "window_mean_Heat_Pump_Heating_Flow_Temperature": "Mean Heat Pump Flow Temperature (°C)",
    },
)

flow_scatter_2.layout.update(showlegend=False)
flow_scatter_2.update_traces(opacity=scatter_opacity)
flow_scatter_2.show()


# The value lies below 0.05, showing the relationship to be statistically significant.

# In[ ]:


# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    flow_temp_data[flow_temp_data["HP_Type_2"] == "ASHPs"]["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    flow_temp_data[flow_temp_data["HP_Type_2"] == "ASHPs"]["SPF_value"],
)

round(
    pd.DataFrame(
        {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
        },
        index=[0],
    ).T.rename({0: "value"}, axis=1),
    2,
)


# Below is the empirical distribution of mean flow temperature for LT and HT ASHPs.

# In[ ]:


# Optional code

# The empirical cumulative distribution function further illustrates the shift in temperature range between HT_ASHPs vs LT_ASHPs. HT_ASHPs tend to operate at a higher mean flow temperature.
fig_ecdf = px.ecdf(
    flow_temp_data,
    x="window_mean_Heat_Pump_Heating_Flow_Temperature",
    color="HP_Type",
    height=500,
    width=600,
    title=f"Empirical Distribution",
    labels={
        "SPF_value": "SPFH4",
        "window_mean_Heat_Pump_Heating_Flow_Temperature": "Mean Heat Pump Flow Temperature",
    },
)
fig_ecdf.show()


# In[ ]:


# Count of heat pumps part of flow temp analysis
pd.DataFrame(flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"].groupby("HP_Type")["Property_ID"].count())


# In[ ]:


flow_temp_data["flow_temp_binned"] = pd.cut(
    flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    bins=[25, 35, 45, 55],
)
flow_temp_data = flow_temp_data.sort_values("flow_temp_binned")

flow_temp_hist_labels = (flow_temp_data.groupby("flow_temp_binned")["Property_ID"].count() / 3).astype(int)

flow_temp_hist_df = flow_temp_data[flow_temp_data["SPF_type"] == "SPFH4"]

fig = px.histogram(
    flow_temp_hist_df,
    x=flow_temp_hist_df["flow_temp_binned"].astype(str),
    y="SPF_value",
    width=600,
    height=400,
    labels={"x": "Flow temperature binned"},
    histfunc="avg",
)

fig.add_trace(
    go.Line(
        x=flow_temp_hist_labels.index.astype(str),
        y=[0.1] * flow_temp_hist_labels.index.size,
        mode="markers+text",
        text="n=" + flow_temp_hist_labels.map(str),
        textposition="top center",
        textfont_color="white",
        marker_opacity=0,
        textfont_size=12,
    )
)
fig.update_layout(showlegend=False)
fig.update_yaxes(title="Mean SPFH4")
fig.show()


# It is not appropriate to conclude that HT_ASHPs are fundamentally more efficient than LT_ASHPs. Rather that other factors e.g. heat pump design and refrigerant can have a bigger impact on real world heat pump performance versus the maximum temperature that a heat pump can operate at. This is likely because most HT_ASHPs do not spend much time operating at or close to their maximum temperature.
# 

# The scatter below shows the relationship between flow temperature and SPFH4, split by brand. This chart shares resemblance with the scatter chart of refrigerants above; this is due to Daikin primarily using the R410A refrigerant.

# In[ ]:


# Optional code

# Generate scatter plot to show relationship between mean flow temperature and SPFH4
flow_scatter_3 = px.scatter(
    flow_temp_data,
    x="window_mean_Heat_Pump_Heating_Flow_Temperature",
    y="SPF_value",
    color="HP_Brand",
    width=700,
    trendline="ols",
    title="Relationship between mean flow temperature and SPFH4 by Brand",
    labels={
        "SPF_value": "SPFH4",
        "window_mean_Heat_Pump_Heating_Flow_Temperature": "Mean Heat Pump Flow Temperature",
    },
)

flow_scatter_3.update_traces(opacity=scatter_opacity)
flow_scatter_3.show()


# Frequency distribution of binned flow temperature; SPFH4 value, by heat pump type.

# In[ ]:


# optional code

# The chart below also articulates the significant difference in SPF between the heat pump types,
# especially prominent in the [33,40] flow temperature range.
flow_temp_data["mean_flow_bins"] = pd.cut(
    flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    bins=10,
    precision=1,
)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.update_layout(barmode="group", bargap=0.1, bargroupgap=0)
for hp_type in ["LT_ASHP", "HT_ASHP"]:
    fig.add_trace(
        go.Scatter(
            x=flow_temp_data["mean_flow_bins"].unique().sort_values().astype(str),
            y=flow_temp_data[flow_temp_data["HP_Type"] == hp_type].groupby("mean_flow_bins").mean()["SPF_value"],
            mode="lines",
            name=f"{hp_type} SPFH4",
            text=flow_temp_data[flow_temp_data["HP_Type"] == hp_type]
            .groupby("mean_flow_bins")
            .mean()["SPF_value"]
            .dropna(),
            textposition="top center",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=flow_temp_data["mean_flow_bins"].unique().sort_values().astype(str),
            y=flow_temp_data[flow_temp_data["HP_Type"] == hp_type]
            .groupby("mean_flow_bins")
            .count()["Property_ID"],
            name=f"{hp_type} Count",
        ),
        secondary_y=False,
    )

fig.update_xaxes(title_text="Binned Heat Pump Flow Temperature")
fig.update_yaxes(title_text="Count", secondary_y=False)
fig.update_yaxes(title_text="SPFH4", secondary_y=True)
fig.update_layout(
    width=1000,
    title="Frequency distribution of flow temperature; SPFH4 value, by binned flow temperature",
)

for hp_type, colour in zip(["LT_ASHP", "HT_ASHP"], [cf.colour1_rgb, cf.colour2_rgb]):
    print(colour)
    fig.update_traces(line=dict(color=f"rgb{colour}"), selector=dict(name=f"{hp_type} SPFH4"))
    fig.update_traces(marker=dict(color=f"rgb{colour}"), selector=dict(name=f"{hp_type} Count"))

fig.show()


# Frequency distribution of binned flow temperature; SPFH4 value, by heat pump brand.

# In[ ]:


# Optional code

# The chart below confirms that there is a significant difference in SPF between the different heat pump brands.
flow_temp_data["mean_flow_bins"] = pd.cut(
    flow_temp_data["window_mean_Heat_Pump_Heating_Flow_Temperature"],
    bins=10,
    precision=1,
)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.update_layout(barmode="group", bargap=0.1, bargroupgap=0)
for hp_brand in flow_temp_data["HP_Brand"].unique():
    fig.add_trace(
        go.Scatter(
            x=flow_temp_data["mean_flow_bins"].unique().sort_values().astype(str),
            y=flow_temp_data[flow_temp_data["HP_Brand"] == hp_brand].groupby("mean_flow_bins").mean()["SPF_value"],
            mode="lines",
            name=f"{hp_brand} SPFH4",
            text=flow_temp_data[flow_temp_data["HP_Brand"] == hp_brand]
            .groupby("mean_flow_bins")
            .mean()["SPF_value"]
            .dropna(),
            textposition="top center",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=flow_temp_data["mean_flow_bins"].unique().sort_values().astype(str),
            y=flow_temp_data[flow_temp_data["HP_Brand"] == hp_brand]
            .groupby("mean_flow_bins")
            .count()["Property_ID"],
            name=f"{hp_brand} Count",
        ),
        secondary_y=False,
    )

fig.update_xaxes(title_text="binned heating flow temperature")
fig.update_yaxes(title_text="Count", secondary_y=False)
fig.update_yaxes(title_text="SPFH4", secondary_y=True)
fig.update_layout(
    width=1000,
    title="Frequency distribution of flow temperature; SPFH4 value, by binned flow temperature",
)

fig.update_traces(line=dict(color=f"rgb{cf.colour1_rgb}"), selector=dict(name="Mitsubishi SPFH4"))
fig.update_traces(line=dict(color=f"rgb{cf.colour2_rgb}"), selector=dict(name="Vaillant SPFH4"))
fig.update_traces(line=dict(color=f"rgb{cf.colour3_rgb}"), selector=dict(name="Daikin SPFH4"))
fig.update_traces(marker=dict(color=f"rgb{cf.colour1_rgb}"), selector=dict(name="Mitsubishi Count"))
fig.update_traces(marker=dict(color=f"rgb{cf.colour2_rgb}"), selector=dict(name="Vaillant Count"))
fig.update_traces(marker=dict(color=f"rgb{cf.colour3_rgb}"), selector=dict(name="Daikin Count"))

fig.show()


# The chart below shows the makeup of heat pump brands by HP_Type.

# In[ ]:


# Optional code

# As suspected, this could be largely due to HT_ASHPs being almost entirely Vaillant branded
# and costing considerably more than Mitsubishi and Daikin.
# The chart below illustrates the split of HP_Brands by HP_Type.
px.histogram(flow_temp_data, x="HP_Type", color="HP_Brand", width=500, height=400).show()

# The chart below illustrates the split of heat pump cost by HP_Brand.
# The Vaillant brand is far more expensive on average.
px.histogram(flow_temp_data, x="Cost_HP", color="HP_Brand", nbins=10, width=700)


# The chart below shows the makeup of refrigerants by HP_Type.

# In[ ]:


# Optional code

# The chart below illustrates the split of HP_Brands by HP_Type.
px.histogram(flow_temp_data, x="HP_Type", color="HP_Refrigerant", width=500, height=400).show()

# The chart below illustrates the split of heat pump cost by HP_Brand.
px.histogram(flow_temp_data, x="Cost_HP", color="HP_Refrigerant", nbins=10, width=700)


# #### How do the coldest day and coldest half hour affect SPFH4?
# 
# The typical performance of ASHPs in very cold situations is in the range [2.51, 2.60] when looking at the coldest day and [2.41, 2.55] when looking at the coldest half hour.

# In[ ]:


# Set analysis thresholds
cold_start_month = 11
cold_end_month = 3
cold_min_SPF = 0.75
cold_max_SPF = 7.5

# Calcuate internal/external temperature difference
home_summary_cold["temp_difference"] = (
    home_summary_cold["Coldest_day_Internal_Air_Temperature: mean"]
    - home_summary_cold["Coldest_day_External_Air_Temperature: mean"]
)

# Bin internal/external temperature difference
home_summary_cold["temp_difference_binned"] = (
    pd.cut(home_summary_cold["temp_difference"], bins=10).dropna().astype(str)
)

print(f"all: {len(home_summary_cold)/6}")

# Filter for ASHPs only
home_summary_cold = home_summary_cold[home_summary_cold["HP_Type_2"] == "ASHPs"]


print(f"all_ashps_only: {len(home_summary_cold)/6}")

# Create DataFrame for cold day analysis and filter between analysis thresholds
home_summary_cold_day = home_summary_cold[home_summary_cold["SPF_type"] == "Coldest_day_SPFH4"]

# Filter between analysis thresholds
home_summary_cold_day = home_summary_cold_day[
    (home_summary_cold_day["SPF_value"] <= cold_max_SPF) & (home_summary_cold_day["SPF_value"] >= cold_min_SPF)
]

print(f"day_inside_SPF_threshold: {len(home_summary_cold_day)}")

home_summary_cold_day = home_summary_cold_day[
    (pd.DatetimeIndex(home_summary_cold_day["Coldest_day_start"]).month >= cold_start_month)
    | (pd.DatetimeIndex(home_summary_cold_day["Coldest_day_start"]).month <= cold_end_month)
]

print(f"day_inside_cold_months: {len(home_summary_cold_day)}")

# Rename SPF to COP as COP is no longer seasonal
home_summary_cold_day = home_summary_cold_day.replace({"Coldest_day_SPFH4": "Coldest_day_COP"})

# Rename field so that we can combine day and HH DataFrames
home_summary_cold_day = home_summary_cold_day.rename(
    {"Coldest_day_External_Air_Temperature: mean": "Coldest_External_Air_Temperature: mean"},
    axis=1,
)

# Calculate heating demand output
home_summary_cold_day["Cold_Heating_Demand_Output"] = home_summary_cold_day[
    "Coldest_day_Heat_Pump_Energy_Output"
] + home_summary_cold_day["Coldest_day_Boiler_Energy_Output"].fillna(0)

# Bin external air temperature
home_summary_cold_day["Coldest_External_Air_Temperature_Binned"] = pd.cut(
    home_summary_cold_day["Coldest_External_Air_Temperature: mean"].sort_values(),
    bins=range(-20, 10, 1),
)

# Calculate IQR, lower fence, upper fence for day heating demand output
day_iqr_Cold_Heating_Demand_Output = stats.iqr(home_summary_cold_day["Cold_Heating_Demand_Output"])
# Set fences to +/- 2 *IQR
day_lower = max(
    0,
    (np.percentile(home_summary_cold_day["Cold_Heating_Demand_Output"], 25) - 2 * day_iqr_Cold_Heating_Demand_Output),
)
day_upper = (
    np.percentile(home_summary_cold_day["Cold_Heating_Demand_Output"], 75) + 2 * day_iqr_Cold_Heating_Demand_Output
)

# Flag day heating demand output outliers and remove them
home_summary_cold_day["Cold_Heating_Demand_Output_day_outlier"] = (
    home_summary_cold_day["Cold_Heating_Demand_Output"] < day_lower
) | (home_summary_cold_day["Cold_Heating_Demand_Output"] > day_upper)

home_summary_cold_day = home_summary_cold_day[home_summary_cold_day["Cold_Heating_Demand_Output_day_outlier"] == False]

print(f"day_inside_heating_demand_threshold: {len(home_summary_cold_day)}")

###### HH

# Create DataFrame for cold HH analysis and filter between analysis thresholds
home_summary_cold_HH = home_summary_cold[home_summary_cold["SPF_type"] == "Coldest_HH_SPFH4"]

# Filter between analysis thresholds
home_summary_cold_HH = home_summary_cold_HH[
    (home_summary_cold_HH["SPF_value"] <= cold_max_SPF) & (home_summary_cold_HH["SPF_value"] >= cold_min_SPF)
]

print(f"HH_inside_SPF_threshold: {len(home_summary_cold_HH)}")

home_summary_cold_HH = home_summary_cold_HH[
    (pd.DatetimeIndex(home_summary_cold_HH["Coldest_HH_start"]).month >= cold_start_month)
    | (pd.DatetimeIndex(home_summary_cold_HH["Coldest_HH_start"]).month <= cold_end_month)
]

print(f"HH_inside_cold_months: {len(home_summary_cold_HH)}")

# Rename SPF to COP as COP is no longer seasonal
home_summary_cold_HH = home_summary_cold_HH.replace({"Coldest_HH_SPFH4": "Coldest_HH_COP"})

# Rename field so that we can combine day and HH DataFrames
home_summary_cold_HH = home_summary_cold_HH.rename(
    {"Coldest_HH_External_Air_Temperature: mean": "Coldest_External_Air_Temperature: mean"},
    axis=1,
)

# Calculate heating demand output
home_summary_cold_HH["Cold_Heating_Demand_Output"] = home_summary_cold_HH[
    "Coldest_HH_Heat_Pump_Energy_Output"
] + home_summary_cold_HH["Coldest_HH_Boiler_Energy_Output"].fillna(0)

# Bin external air temperature
home_summary_cold_HH["Coldest_External_Air_Temperature_Binned"] = pd.cut(
    home_summary_cold_HH["Coldest_External_Air_Temperature: mean"].sort_values(),
    bins=range(-20, 10, 1),
)


# Calculate IQR, lower fence, upper fence for HH heating demand output
HH_iqr_Cold_Heating_Demand_Output = stats.iqr(home_summary_cold_HH["Cold_Heating_Demand_Output"])

# Set fences to +/- 2 *IQR
HH_lower = max(
    0, (np.percentile(home_summary_cold_HH["Cold_Heating_Demand_Output"], 25) - 2 * HH_iqr_Cold_Heating_Demand_Output)
)
HH_upper = (
    np.percentile(home_summary_cold_HH["Cold_Heating_Demand_Output"], 75) + 2 * HH_iqr_Cold_Heating_Demand_Output
)

# Flag HH heating demand output outliers and remove them
home_summary_cold_HH["Cold_Heating_Demand_Output_HH_outlier"] = (
    home_summary_cold_HH["Cold_Heating_Demand_Output"] < HH_lower
) | (home_summary_cold_HH["Cold_Heating_Demand_Output"] > HH_upper)

home_summary_cold_HH = home_summary_cold_HH[home_summary_cold_HH["Cold_Heating_Demand_Output_HH_outlier"] == False]

print(f"HH_inside_heating_demand_threshold: {len(home_summary_cold_HH)}")

# Combine day and HH DataFrames
home_summary_cold_all = pd.concat([home_summary_cold_day, home_summary_cold_HH])

print(f"day lower: {day_lower}")
print(f"day upper: {day_upper}")
print(f"HH lower: {HH_lower}")
print(f"HH upper: {HH_upper}")


# In[ ]:


# Generate stats for coldest day
stats_coldest_day = af.stats_by_hp_and_SPF_type(home_summary_cold_day.dropna(subset="SPF_value"), by="HP_Type_2")
stats_coldest_day["Mean external temp"] = round(
    home_summary_cold_day["Coldest_External_Air_Temperature: mean"].mean(), 2
)

stats_coldest_day[
    ["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)", "Mean external temp"]
].reset_index(drop=True).set_index(["HP_Type_2", "SPF_type"])


# In[ ]:


home_summary_cold_day["Coldest_External_Air_Temperature: mean"].mean()


# In[ ]:


# Generate stats for coldest HH
stats_coldest_HH = af.stats_by_hp_and_SPF_type(home_summary_cold_HH.dropna(subset="SPF_value"), by="HP_Type_2")
stats_coldest_HH["Mean external temp"] = round(
    home_summary_cold_HH["Coldest_External_Air_Temperature: mean"].mean(), 2
)

stats_coldest_HH[
    ["HP_Type_2", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)", "Mean external temp"]
].reset_index(drop=True).set_index(["HP_Type_2", "SPF_type"])


# Frequency distribution of day external temperature binned.

# In[ ]:


px.histogram(
    home_summary_cold_day,
    x=home_summary_cold_day["Coldest_External_Air_Temperature_Binned"].sort_values().astype(str),
    width=700,
    labels={"x": "External Temperature Binned"},
    title="Coldest day binned external temperature",
)


# Frequency distribution of HH external temperature binned.

# In[ ]:


px.histogram(
    home_summary_cold_HH,
    x=home_summary_cold_HH["Coldest_External_Air_Temperature_Binned"].sort_values().astype(str),
    width=700,
    labels={"x": "External Temperature Binned"},
    title="Coldest HH binned external temperature",
)


# The chart below shows the relationship between external air temperature and COP for the coldest day and HH. As expected, the the lower the external air temperature, the lower the COP observed.

# In[ ]:


fig = px.scatter(
    home_summary_cold_all,
    x="Coldest_External_Air_Temperature: mean",
    y="SPF_value",
    color="SPF_type",
    width=800,
    trendline="ols",
    labels={
        "Coldest_External_Air_Temperature: mean": "Coldest External Air Temperature: mean",
        "SPF_value": "COP",
        "SPF_type": "SPF Type",
    },
)
fig.update_traces(opacity=scatter_opacity)

for spf_type, colour in zip(["Coldest_day_COP", "Coldest_HH_COP"], colours[0:2]):

    # Implement linear regression and produce statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        home_summary_cold_all[home_summary_cold_all["SPF_type"] == spf_type]["Coldest_External_Air_Temperature: mean"],
        home_summary_cold_all[home_summary_cold_all["SPF_type"] == spf_type]["SPF_value"],
    )

    print(
        round(
            pd.DataFrame(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "std_err": std_err,
                },
                index=[0],
            ).T.rename({0: "value"}, axis=1),
            2,
        )
    )

    x = home_summary_cold_all[home_summary_cold_all["SPF_type"] == spf_type][
        "Coldest_External_Air_Temperature: mean"
    ].min()
    y = slope * x + intercept

    fig.add_trace(
        go.Scatter(
            x=[x * 1.05],
            y=[y * 0.95],
            mode="lines+text",
            text=f"R²={round(r_value**2,2)}",
            textposition="bottom center",
            textfont=dict(color=f"rgb{colour}"),
        )
    )

fig.update_layout(showlegend=False)


# The value for day COP vs external temperature lies below 0.05, showing the relationship to be statistically significant.

# In[ ]:


# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    home_summary_cold_all[home_summary_cold_all["SPF_type"] == "Coldest_day_COP"][
        "Coldest_External_Air_Temperature: mean"
    ],
    home_summary_cold_all[home_summary_cold_all["SPF_type"] == "Coldest_day_COP"]["SPF_value"],
)

round(
    pd.DataFrame(
        {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
        },
        index=[0],
    ).T.rename({0: "value"}, axis=1),
    2,
)


# The value for HH COP vs external temperature lies below 0.05, showing the relationship to be statistically significant.

# In[ ]:


# optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    home_summary_cold_all[home_summary_cold_all["SPF_type"] == "Coldest_HH_COP"][
        "Coldest_External_Air_Temperature: mean"
    ],
    home_summary_cold_all[home_summary_cold_all["SPF_type"] == "Coldest_HH_COP"]["SPF_value"],
)

round(
    pd.DataFrame(
        {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
        },
        index=[0],
    ).T.rename({0: "value"}, axis=1),
    2,
)


# The chart below shows the relationship between heating demand output and COP for the coldest day.

# In[ ]:


fig = px.scatter(
    home_summary_cold_day,
    x="Cold_Heating_Demand_Output",
    y="SPF_value",
    width=800,
    title="Coldest day",
    trendline="ols",
    labels={"SPF_value": "COP", "Cold_Heating_Demand_Output": "Heating Demand Output"},
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# The chart below shows the relationship between heating demand output and COP for the coldest HH.

# In[ ]:


fig = px.scatter(
    home_summary_cold_HH,
    x="Cold_Heating_Demand_Output",
    y="SPF_value",
    width=800,
    title="Coldest HH",
    trendline="ols",
    labels={"SPF_value": "COP", "Cold_Heating_Demand_Output": "Heating Demand Output"},
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# The top 10 days by their frequency in the data are listed below along with their mean external temperature.

# In[ ]:


coldest_days = (
    home_summary_cold_day.groupby("Coldest_day_start")["Coldest_External_Air_Temperature: mean"]
    .agg(["count", "mean"])
    .sort_values(by="count", ascending=False)
)
coldest_days = coldest_days.rename({"mean": "Mean External Air Temp"}, axis=1)
coldest_days.index = pd.to_datetime(coldest_days.index)
coldest_days.nlargest(10, "count").round(1)


# Mean day external temperature:

# In[ ]:


round(home_summary_cold_day["Coldest_External_Air_Temperature: mean"].mean(), 2)


# Mean HH external temperature:

# In[ ]:


round(home_summary_cold_HH["Coldest_External_Air_Temperature: mean"].mean(), 2)


# The top 10 half hours by their frequency in the data are listed below along with their mean external temperature.

# In[ ]:


coldest_HHs = (
    home_summary_cold_HH.groupby("Coldest_HH_start")["Coldest_External_Air_Temperature: mean"]
    .agg(["count", "mean"])
    .sort_values(by="count", ascending=False)
)
coldest_HHs = coldest_HHs.rename({"mean": "Mean External Air Temp"}, axis=1)
coldest_HHs.nlargest(10, "count")


# The table below shows stats on the day COP for binned internal/external temperature differences.

# In[ ]:


af.stats_by_hp_and_SPF_type(
    home_summary_cold_day.dropna(subset="temp_difference_binned"),
    by="temp_difference_binned",
)[["temp_difference_binned", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index(drop=True).set_index(
    ["temp_difference_binned", "SPF_type"]
)


# The chart below shows the relationship between internal/external temperature and COP.

# In[ ]:


temp_diff_scatter_df = home_summary_cold_day[
    (home_summary_cold_day["temp_difference"] >= 5)
    & (home_summary_cold_day["Coldest_day_Internal_Air_Temperature: mean"] > 5)
]

# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    temp_diff_scatter_df["temp_difference"],
    temp_diff_scatter_df["SPF_value"],
)

print(
    round(
        pd.DataFrame(
            {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
            },
            index=[0],
        ).T.rename({0: "value"}, axis=1),
        2,
    )
)

x = temp_diff_scatter_df["temp_difference"].max()
y = slope * x + intercept

fig = px.scatter(
    temp_diff_scatter_df,
    x="temp_difference",
    y="SPF_value",
    width=800,
    title="COP by internal/external temp difference",
    labels={
        "SPF_value": "Day COP",
        "temp_difference": "Internal to external temperature difference",
    },
    trendline="ols",
)

fig.update_traces(opacity=scatter_opacity)
fig.add_trace(
    go.Scatter(
        x=[x * 1.05],
        y=[y],
        mode="lines+text",
        text=f"R²={round(r_value**2,2)}",
        textposition="middle center",
        textfont=dict(color=f"rgb{list(cf.colour_scheme().values())[0]}"),
    )
)
fig.update_layout(showlegend=False)


# The table below shows stats on the day COP for binned internal/external temperature differences.

# In[ ]:


af.stats_by_hp_and_SPF_type(
    home_summary_cold_HH.dropna(subset="temp_difference_binned"),
    by="temp_difference_binned",
)[["temp_difference_binned", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index(drop=True).set_index(
    ["temp_difference_binned", "SPF_type"]
)


# The chart below shows the relationship between internal/external temperature and COP.

# In[ ]:


temp_diff_scatter_df_HH = home_summary_cold_HH[
    (home_summary_cold_HH["temp_difference"] >= 5)
    & (home_summary_cold_HH["Coldest_day_Internal_Air_Temperature: mean"] > 5)
]

# Optional code

# Implement linear regression and produce statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    temp_diff_scatter_df_HH["temp_difference"],
    temp_diff_scatter_df_HH["SPF_value"],
)

print(
    round(
        pd.DataFrame(
            {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
            },
            index=[0],
        ).T.rename({0: "value"}, axis=1),
        2,
    )
)

x = temp_diff_scatter_df_HH["temp_difference"].max()
y = slope * x + intercept

fig = px.scatter(
    home_summary_cold_HH[
        (home_summary_cold_HH["temp_difference"] >= 5)
        & (home_summary_cold_HH["Coldest_HH_Internal_Air_Temperature: mean"] > 5)
    ],
    x="temp_difference",
    y="SPF_value",
    width=800,
    title="COP by internal/external temp difference",
    labels={"SPF_value": "Half-hour COP"},
    trendline="ols",
)

fig.update_traces(opacity=scatter_opacity)
fig.update_xaxes(title="Internal to external temperature difference")
fig.add_trace(
    go.Scatter(
        x=[x * 1.05],
        y=[y],
        mode="lines+text",
        text=f"R²={round(r_value**2,2)}",
        textposition="middle center",
        textfont=dict(color=f"rgb{list(cf.colour_scheme().values())[0]}"),
    )
)
fig.update_layout(showlegend=False)


# The chart below shows the relationship between internal temperature and day COP.

# In[ ]:


fig = px.scatter(
    home_summary_cold_day,
    x="Coldest_day_Internal_Air_Temperature: mean",
    y="SPF_value",
    width=500,
    title="Day COP by internal temp",
    labels={"SPF_value": "Day COP"},
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# The chart below shows the relationship between internal temperature and HH COP.

# In[ ]:


fig = px.scatter(
    home_summary_cold_HH,
    x="Coldest_HH_Internal_Air_Temperature: mean",
    y="SPF_value",
    width=500,
    title="HH COP by internal temp",
    labels={"SPF_value": "HH COP"},
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# In[ ]:


# For each home, we create a DataFrame that compares the day COP and HH COP

cold_day_HH_comparison = (
    home_summary_cold_day[["Property_ID", "SPF_value"]]
    .merge(home_summary_cold_HH[["Property_ID", "SPF_value"]], on="Property_ID")
    .rename({"SPF_value_x": "Coldest_Day_COP", "SPF_value_y": "Coldest_HH_COP"}, axis=1)
)
cold_day_HH_comparison["Day COP higher than HH COP"] = (
    cold_day_HH_comparison["Coldest_Day_COP"] > cold_day_HH_comparison["Coldest_HH_COP"]
)
cold_day_HH_comparison["COP_diff"] = (
    cold_day_HH_comparison["Coldest_Day_COP"] - cold_day_HH_comparison["Coldest_HH_COP"]
)
cold_day_HH_comparison


# We plot this data to observe the scale at which the day COP is higher than than the HH COP. It is expected that the day COP is higher.

# In[ ]:


fig = px.scatter(
    cold_day_HH_comparison,
    x="Coldest_HH_COP",
    y="Coldest_Day_COP",
    color="Day COP higher than HH COP",
    hover_data=["Property_ID"],
    width=800,
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# #### How does floor area affect SPF?

# In[ ]:


# Bin floor area and calculate average total energy output
homes_within_threshold["Total_Floor_Area"] = pd.to_numeric(homes_within_threshold["Total_Floor_Area"], errors="coerce")
homes_within_threshold["Total_Floor_Area_Binned"] = pd.cut(homes_within_threshold["Total_Floor_Area"], 5, precision=0)
floor_area_binned_df = homes_within_threshold.groupby("Total_Floor_Area_Binned").agg(
    {"window_Total_Energy_Output": "mean", "Property_ID": "count"}
)
pd.DataFrame(floor_area_binned_df)


# The chart below shows the mean total energy output for each floor area bin.

# In[ ]:


# Generate bar chart showing mean total energy output for each floor area bin
px.bar(
    floor_area_binned_df.sort_index(),
    x=floor_area_binned_df.index.astype(str),
    y="window_Total_Energy_Output",
    width=700,
    labels={
        "x": "Total Floor Area Binned",
        "window_Total_Energy_Output": "Total Energy Output",
    },
    title="Mean total energy output by floor area binned",
)


# In[ ]:


# Generate bar chart showing mean total energy output for each floor area bin

floor_area_binned_df["window_Total_Energy_Output"] = floor_area_binned_df["window_Total_Energy_Output"] / 365
floor_area_binned_df["Property_ID"] = (floor_area_binned_df["Property_ID"] / 3).astype(int)
floor_area_binned_df


# In[ ]:


fig = px.bar(
    floor_area_binned_df.sort_index(),
    x=floor_area_binned_df.index.astype(str),
    y="window_Total_Energy_Output",
    width=700,
    labels={
        "x": "Total Floor Area Binned",
        "window_Total_Energy_Output": "Mean kWh/day",
    },
    title="Mean total energy output by floor area binned",
)

fig.add_trace(
    go.Line(
        x=floor_area_binned_df.index.astype(str),
        y=[y_range_min] * floor_area_binned_df.index.size,
        mode="markers+text",
        text="n=" + floor_area_binned_df["Property_ID"].map(str),
        textposition="top center",
        textfont_color="white",
        marker_opacity=0,
        textfont_size=12,
    )
)

fig.update_layout(showlegend=False)
fig.show()


# The chart below shows a strong relationship between total floor are and total energy output.

# In[ ]:


# Calculate confidence interval
stats_by_house_type["half_interval"] = stats_by_house_type["Mean"] - stats_by_house_type["Mean CI Lower"]

y_range_min = 1.8
y_range_max = 3.8

# Generate bar graph of means and confidence intervals for each group
fig = px.bar(
    stats_by_house_type,
    x=stats_by_house_type.index,
    y="Mean",
    width=600,
    range_y=[y_range_min, y_range_max],
    error_y=stats_by_house_type["half_interval"],
    title="ASHPs Mean SPFH4 with 95% Confidence Intervals",
    labels={"Mean": "Mean SPFH4", "House_Type": "House Type"},
    hover_data=["Homes"],
)

fig.add_trace(
    go.Line(
        x=stats_by_house_type.index,
        y=[y_range_min] * stats_by_house_type.index.size,
        mode="markers+text",
        text="n=" + stats_by_house_type["Homes"].map(str),
        textposition="top center",
        textfont_color="white",
        marker_opacity=0,
        textfont_size=10,
    )
)
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


# Generate scatter chart showing mean total energy output versus total floor area for each home
fig = px.scatter(
    homes_within_threshold,
    x="Total_Floor_Area",
    y="window_Total_Energy_Output",
    width=800,
    trendline="ols",
    labels={
        "window_Total_Energy_Output": "Total Energy Output",
        "Total_Floor_Area": "Total Floor Area",
    },
    title="Total floor area vs total energy output",
)
fig.update_traces(opacity=scatter_opacity)
fig.show()


# #### How often do ASHPs operate at higher temperatures (above 65 degrees C)?

# The table below spfh stats for ASHPs split between high temperature operation and not.

# In[ ]:


# Calculate statistics
af.stats_by_hp_and_SPF_type(
    homes_within_threshold[
        (homes_within_threshold["HP_Type_2"] == "ASHPs") & (homes_within_threshold["SPF_type"] != "SPFH3")
    ].dropna(subset="window_flow > 65C"),
    by="window_flow > 65C",
)[["window_flow > 65C", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index(drop=True).set_index(
    ["window_flow > 65C", "SPF_type"]
).sort_values(["window_flow > 65C", "SPF_type"])


# The table below spfh stats for HT_ASHPs only, split between high temperature operation and not. Most HT_ASHPs operate above 65 degrees C within their analysis window.

# In[ ]:


# Calculate statistics
af.stats_by_hp_and_SPF_type(
    homes_within_threshold[
        (homes_within_threshold["HP_Type"] == "HT_ASHP") & (homes_within_threshold["SPF_type"] != "SPFH3")
    ].dropna(subset="window_flow > 65C"),
    by="window_flow > 65C",
)[["window_flow > 65C", "SPF_type", "Homes", "Median (IQR)", "Mean (95% CI)"]].reset_index(drop=True).set_index(
    ["window_flow > 65C", "SPF_type"]
).sort_index(
    ascending=False
)


# In[ ]:


# Bin window_% > 65C field
homes_within_threshold["window_65_deg_binned"] = pd.cut(
    homes_within_threshold["window_% > 65C"].dropna().sort_values(),
    bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
)

# Filter for HT_ASHP and SPFH4 only
homes_within_threshold_65_deg_binned = homes_within_threshold[
    (homes_within_threshold["HP_Type"] == "HT_ASHP") & (homes_within_threshold["SPF_type"] == "SPFH3")
].dropna(subset="window_65_deg_binned")

# Sorting and formatting of binned column
homes_within_threshold_65_deg_binned = homes_within_threshold_65_deg_binned.sort_values("window_65_deg_binned").astype(
    str
)
homes_within_threshold_65_deg_binned["window_65_deg_binned"] = homes_within_threshold_65_deg_binned[
    "window_65_deg_binned"
].replace(
    {
        "(0.0, 0.01]": "(0, 1%]",
        "(0.01, 0.02]": "(1, 2%]",
        "(0.02, 0.03]": "(2, 3%]",
        "(0.03, 0.04]": "(3, 4%]",
        "(0.04, 0.05]": "(4, 5%]",
        "(0.05, 0.06]": "(5, 6%]",
        "(0.06, 0.07]": "(6, 7%]",
        "(0.07, 0.08]": "(7, 8%]",
        "(0.08, 0.09]": "(8, 9%]",
    }
)


# The chart belows shows that most HT_ASHPs only operate above 65 degrees C for less than 1% of the time.

# In[ ]:


# Generate histogram for binned % of window above 65 degrees C
fig = px.histogram(
    homes_within_threshold_65_deg_binned,
    x="window_65_deg_binned",
    y="SPF_value",
    width=800,
    labels={
        "window_65_deg_binned": "% of window above 65degC",
        "spf_value": "SPFH3",
        "(0.0, 0.01]": "A",
    },
    histfunc="count",
    title="SPF by window_% > 65C",
)

fig.update_layout(xaxis_tickformat="0%")
fig.show()


# #### Hybrid Heat Pumps - Heat Pump versus Boiler Analysis
# The section below looks only at Hybrid Heat Pumps.
# 
# The proportion of the total heat energy output that comes from the heat pump (as opposed to the boiler) is calculated as 'hp_proportion_of_output'.
# 
# ##### What proportion of heat demand is supplied by the heat pump in a hybrid system?

# In[ ]:


# Filter for hybrid units
home_summary_hybrids = homes_within_threshold[homes_within_threshold["HP_Type"] == "Hybrid"]


# In[ ]:


# Filter for SPFH4 only
homes_SPFH4_hybrids = home_summary_hybrids[home_summary_hybrids["SPF_type"] == "SPFH4"]

# Filter for columns of interest
homes_SPFH4_hybrids = homes_SPFH4_hybrids[
    [
        "Property_ID",
        "HP_Type",
        "SPF_type",
        "window_Heat_Pump_Energy_Output",
        "window_Boiler_Energy_Output",
        "SPF_value",
        "Removed_from_analysis",
    ]
]

# Calculate the proportion of total energy output that the heat pump provides
homes_SPFH4_hybrids["hp_proportion_of_output"] = homes_SPFH4_hybrids[
    "window_Heat_Pump_Energy_Output"
] / homes_SPFH4_hybrids[["window_Heat_Pump_Energy_Output", "window_Boiler_Energy_Output"]].sum(axis=1)

# Proportion values <0 and >0.99 have been removed as they are erroneous or outliers
homes_SPFH4_hybrids = homes_SPFH4_hybrids[
    (homes_SPFH4_hybrids["Removed_from_analysis"] == False)
    & (homes_SPFH4_hybrids["hp_proportion_of_output"] >= 0)
    & (homes_SPFH4_hybrids["hp_proportion_of_output"] <= 0.99)
]


# The chart below shows the frequency distribution of binned heat pump energy output as a percentage of total output.

# In[ ]:


# Generate frequency distribution across binned heat pump energy output as percentage of total output

hist_proportion_freq = px.histogram(
    homes_SPFH4_hybrids,
    x="hp_proportion_of_output",
    nbins=15,
    width=800,
    labels={"hp_proportion_of_output": "Heat pump output as percentage of total space heating output"},
    title="Frequency Distribution by Proportion of Heat Pump Usage",
)
hist_proportion_freq.update_layout(bargap=0.05, showlegend=False, xaxis_tickformat="0%")
hist_proportion_freq.show()


# The table below shows the SPFH4 values for hybrid heat pumps.

# In[ ]:


# Calculate statistics for hybrids
af.stats_by_hp_and_SPF_type(homes_SPFH4_hybrids)[
    ["HP_Type", "SPF_type", "Homes", "Mean (95% CI)", "Median (IQR)"]
].set_index(["HP_Type", "SPF_type"])


# ##### Is heat pump efficiency higher if the heat pump provides more of the heat demand?

# There is a statistically significant relationship (p<0.05) between proportion_hp_to_boiler and SPF – systems with higher heat pump usage tend to have lower SPFs. Note that there is a lot of variation in performance that is not explained by this factor (R2=0.29).

# In[ ]:


# optional code

# To this relationship, a linear model provides the following statistics:
proportion_median = homes_SPFH4_hybrids["hp_proportion_of_output"].median()
ashp_median_SPFH2 = homes_within_threshold["SPF_value"][
    (homes_within_threshold["HP_Type"] == "ASHP") & (homes_within_threshold["SPF_type"] == "SPFH4")
].median()

slope, intercept, r_value, p_value, std_err = stats.linregress(
    homes_SPFH4_hybrids["hp_proportion_of_output"], homes_SPFH4_hybrids["SPF_value"]
)

print(
    round(
        pd.DataFrame(
            {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
            },
            index=[0],
        ).T.rename({0: "value"}, axis=1),
        2,
    )
)

x = homes_SPFH4_hybrids["hp_proportion_of_output"].max()
y = slope * x + intercept

# Generate scatter plot and regression line
scatter_hpboiler_vs_SPF = px.scatter(
    homes_SPFH4_hybrids,
    x="hp_proportion_of_output",
    y="SPF_value",
    trendline="ols",
    width=600,
    title="Relationship between heat pump proportion and SPFH4",
    labels={
        "SPF_value": "SPFH4",
        "hp_proportion_of_output": "Heat pump output as percentage of total space heating output",
    },
)
scatter_hpboiler_vs_SPF.update_traces(opacity=scatter_opacity)
scatter_hpboiler_vs_SPF.update_layout(xaxis_tickformat="0%")
scatter_hpboiler_vs_SPF.add_trace(
    go.Scatter(
        x=[x * 0.9],
        y=[y],
        mode="lines+text",
        text=f"R²={round(r_value**2,2)}",
        textposition="middle center",
        textfont=dict(color=f"rgb{list(cf.colour_scheme().values())[0]}"),
    )
)
scatter_hpboiler_vs_SPF.update_layout(showlegend=False)


# The table below shows statistics on the heat pump proportion of total energy output amongst hybrid heat pumps.

# In[ ]:


pd.DataFrame(homes_SPFH4_hybrids["hp_proportion_of_output"].describe())

