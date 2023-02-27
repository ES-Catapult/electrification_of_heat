import os
import pandas as pd
import requests
import json
import numpy as np
import scipy.stats as stats


def reformat_SPF_to_long(data: pd.DataFrame, cold_analysis: bool) -> pd.DataFrame:
    """Converts the spfh_analysis csv from 1 row per home to 1 row per home per and per spf_type

    Args:
        data (pd.DataFrame): dataframe with one row per home (spfh_analysis.csv)
        cold_analysis (bool): Whether to also create a row for each spf_type in the coldest day and coldest half hour

    Returns:
        pd.DataFrame: _description_
    """

    # Select columns which do not include SPF in their names
    homes_output_info = data.loc[:, ~data.columns.str.contains("SPF")]

    # Select columns which do include SPF in their names
    homes_output_SPFHx = data.filter(like="SPF", axis=1).copy()

    # Add site reference id to homes_output_info
    homes_output_SPFHx["Property_ID"] = homes_output_info["Property_ID"]

    # Select which columns to use for the melt, depending on the input provided in the cold_analysis arg
    if cold_analysis == True:
        value_vars = tuple(
            [
                "SPFH2",
                "SPFH3",
                "SPFH4",
                "Coldest_day_SPFH2",
                "Coldest_day_SPFH3",
                "Coldest_day_SPFH4",
                "Coldest_HH_SPFH2",
                "Coldest_HH_SPFH3",
                "Coldest_HH_SPFH4",
            ],
        )
        # Make sure start dates are in correct datetime format
        for cold_interval_start in ["Coldest_day_start", "Coldest_HH_start"]:
            data[cold_interval_start] = pd.to_datetime(data[cold_interval_start], errors="coerce")

    else:
        value_vars = tuple(
            [
                "SPFH2",
                "SPFH3",
                "SPFH4",
            ],
        )

    # Perform the melt
    homes_output_SPFHx = homes_output_SPFHx.melt(
        id_vars="Property_ID",
        value_vars=value_vars,
        var_name="SPF_type",
        value_name="SPF_value",
    )

    # Merge back the non-SPF columns
    homes_output = homes_output_info.merge(homes_output_SPFHx, on="Property_ID")
    homes_output = homes_output.sort_values("Property_ID")

    return homes_output


def add_anonymised_ids(data: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Gets the anonymised home ids and joins them onto the SPF_data by home

    Args:
        data (pd.DataFrame): SPF_data per home and SPF_type
        file_path (str): file path for the home anonymisation lookup
    Returns:
        pd.DataFrame: DataFrame with relevant fields added
    """
    anonymised_ids = pd.read_excel(
        "S:\Projects\Electrification of Heat\WP3 - Data Collection & Co-ordination\Property Number Anonymisation.xlsx"
    )

    data["Property_ID"] = data["Property_ID"].astype(str)
    anonymised_ids["House_ID"] = anonymised_ids["House_ID"].astype(str)

    data = data.merge(
        anonymised_ids,
        how="left",
        left_on="Property_ID",
        right_on="House_ID",
        copy=False,
        suffixes=(None, "_y"),
    )

    data = data.loc[:, ~data.columns.str.endswith("_y")]
    data = data.drop(["House_ID"], axis=1)

    return data


def stats_by_hp_and_SPF_type(data: pd.DataFrame, by: str = "HP_Type", rounding: int = 2) -> pd.DataFrame:
    """Provides a variety of stats for SPF_value by SPF_type and another specified split_by type.
    Includes 95% confidence interval for the mean and IQR for the median.

    Args:
        data (pd.DataFrame): Dataframe with 1 row per home and per SPF_type

    Returns:
        pd.DataFrame: Table of statistics
    """
    stats_table = pd.DataFrame()
    rounding = rounding

    for by_type in data[by].unique():
        homes_output_hp = data[data[by] == by_type]
        for SPF_type in data["SPF_type"].unique():
            homes_output_split_SPF_type = homes_output_hp[homes_output_hp["SPF_type"] == SPF_type]
            mean = np.mean(homes_output_split_SPF_type["SPF_value"])
            mean_confidence = stats.t.interval(
                alpha=0.95,
                df=len(homes_output_split_SPF_type) - 1,
                loc=np.mean(homes_output_split_SPF_type["SPF_value"]),
                scale=stats.sem(homes_output_split_SPF_type["SPF_value"]),
            )
            stats_table_part = pd.DataFrame(mean_confidence).T
            stats_table_part[by] = by_type
            stats_table_part["SPF_type"] = SPF_type
            stats_table_part["Homes"] = homes_output_split_SPF_type["Property_ID"].unique().size
            stats_table_part["Min"] = np.min(homes_output_split_SPF_type["SPF_value"])
            stats_table_part["Max"] = np.max(homes_output_split_SPF_type["SPF_value"])
            stats_table_part["Std"] = np.std(homes_output_split_SPF_type["SPF_value"])
            stats_table_part["Mean"] = mean
            stats_table_part["Median"] = np.median(homes_output_split_SPF_type["SPF_value"])
            stats_table_part["Mean (95% CI)"] = str(np.round(mean, 2)) + " " + str(np.round(mean_confidence, rounding))
            stats_table_part["Median (IQR)"] = (
                str(np.round(np.median(homes_output_split_SPF_type["SPF_value"]), rounding))
                + " ["
                + str(
                    np.round(
                        np.percentile(homes_output_split_SPF_type["SPF_value"], 25),
                        rounding,
                    )
                )
                + " "
                + str(
                    np.round(
                        np.percentile(homes_output_split_SPF_type["SPF_value"], 75),
                        rounding,
                    )
                )
                + "]"
            )

            # remove un-needed columns used to create mean confidence interval

            stats_table_part = stats_table_part.rename({0: "Mean CI Lower", 1: "Mean CI Upper"}, axis=1)
            # stats_table_part = stats_table_part.drop([0, 1], axis=1)
            stats_table = pd.concat([stats_table, stats_table_part])

    stats_table = pd.DataFrame(stats_table)
    stats_table = stats_table.reset_index(drop=True)

    return stats_table


def remove_homes_without_all_SPF_types(data: pd.DataFrame) -> pd.DataFrame:
    """Removes the home from the dataset if either SPFH2, SPFH3 or SPFH4 are not present

    Args:
        data (pd.DataFrame): home_summary dataframe

    Returns:
        pd.DataFrame: homes_output with any homes that required removal
    """
    homes_out_of_threshold = data.groupby("Property_ID")["SPF_type"].count() < 3
    homes_out_of_threshold = homes_out_of_threshold[homes_out_of_threshold == True].reset_index()
    homes_out_of_threshold["Property_ID"].unique()
    homes_output_within_threshold = data[~(data["Property_ID"].isin(homes_out_of_threshold["Property_ID"].unique()))]

    return homes_output_within_threshold


def filter_homes_within_threshold(
    data: pd.DataFrame,
    SPF_value_min_threshold: float,
    SPF_value_max_threshold: float,
    window_max_score_threshold: int,
) -> pd.DataFrame:
    """Uses the thresholds provided to filter the dataframe

    Args:
        data (pd.DataFrame): Dataframe with information on homes and their performance  to be filtered
        SPF_value_min_threshold (float): Minimum acceptable value for all SPF types
        SPF_value_max_threshold (float): Maximum acceptable value for all SPF types
        window_max_score_threshold (int): The data quality score threshold that the data must not exceed

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    data["within_threshold"] = (
        (data["SPF_value"] >= SPF_value_min_threshold)
        & (data["SPF_value"] <= SPF_value_max_threshold)
        & (data["window_max_score"] < window_max_score_threshold)
    )

    homes_output_within_threshold = data[data["within_threshold"] == True]

    homes_output_within_threshold = remove_homes_without_all_SPF_types(homes_output_within_threshold)

    return homes_output_within_threshold


def add_supplementary_data(data: pd.DataFrame) -> pd.DataFrame:
    """Adds supplementary data from usmart. Joins on Property_ID.

    Args:
        data (pd.DataFrame): Dataframe with data to be supplemented

    Returns:
        pd.DataFrame: Dataframe with data supplemented
    """
    urls = [
        "https://api.usmart.io/org/92610836-6c2b-4a26-a0a0-b903bde0dc46/231dd812-ed7f-41b4-bbe2-c0929ca95299/latest/urql?limit(-1)"
    ]

    headers = {
        "cache-control": "no-cache",
        "api-key-id": os.environ["USMART_KEY_ID"],
        "api-key-secret": os.environ["USMART_KEY_SECRET"],
    }

    db = []
    for url in urls:
        response = requests.request("GET", url, headers=headers)
        file = pd.DataFrame(json.loads(response.text))
        db.append(file)
    df = pd.concat(db)

    df["MCS_Total_Energy"] = df["MCS_SHAnnual"] + df["MCS_DHWAnnual"] + df["MCS_Immersion"]

    data = data.merge(
        df[
            [
                "Property_ID",
                "Delivery_Contractor",
                "HouseIncome",
                "Social_Group",
                "EmployType",
                "House_Form",
                "House_Age",
                "Total_Floor_Area",
                "HP_Size_kW",
                "Name_Install",
                "HP_Installed",
                "HP_Brand",
                "HP_Model",
                "Cost_HP",
                "MCS_SHLoad",
                "Postcode_1",
                "HP_Refrigerant",
            ]
        ],
        on="Property_ID",
        how="left",
        copy=False,
        suffixes=(None, "_y"),
    )

    data = data.loc[:, ~data.columns.str.endswith("_y")]

    data = data.rename(
        {
            "HP_Installed": "HP_Type",
            "HouseIncome": "House_Income",
            "EmployType": "Employ_Type",
        },
        axis=1,
    )

    data["HP_Type"] = data["HP_Type"].replace({"ASHP": "LT_ASHP"})
    data["HP_Type_2"] = data["HP_Type"].replace({"LT_ASHP": "ASHPs", "HT_ASHP": "ASHPs"})
    data["HP_Type_3"] = data["HP_Type_2"].replace({"ASHPs": "All_ASHPs", "Hybrid": "All_ASHPs"})

    data["relative_sizing"] = pd.to_numeric(data["HP_Size_kW"], errors="coerce") - pd.to_numeric(
        data["MCS_SHLoad"], errors="coerce"
    )

    data = data.replace(
        {
            "End -Terrace": "End-Terrace",
            "End - Terrace": "End-Terrace",
            "Mid - Terrace": "Mid-Terrace",
            "Semi-Detatched": "Semi-Detached",
        }
    )

    data["House_Form"] = data["House_Form"].fillna("Unknown")
    data["House_Form"] = data["House_Form"].replace("", "Unknown")

    data["House_Age"] = data["House_Age"].fillna("Unknown")
    data["House_Age"] = data["House_Age"].replace("", "Unknown")

    data["Property_ID"] = data["Property_ID"].fillna("Unknown")
    data["Property_ID"] = data["Property_ID"].replace("", "Unknown")

    data = data.rename({"House_Age": "House_Age_Raw"}, axis=1)
    data["House_Age_Cleaned"] = data["House_Age_Raw"].replace(
        {
            "1900 - 1929": "Pre-1919",
            "1930 - 1949": "1919-1944",
            "1950 - 1966": "1945-1964",
            "1967 - 1975": "1965-1980",
            "1976 - 1982": "1965-1980",
            "1983 - 1990": "1981-1990",
            "1991 - 1995": "1991-2000",
            "1996 - 2002": "1991-2000",
            "2003 - 2006": "2001+",
            "2007 - 2011": "2001+",
            "2012 onwards": "2001+",
            "Before 1900": "Pre-1919",
            "Pre 1919": "Pre-1919",
            "null": "Unknown",
        }
    )

    data.columns = data.columns.str.replace("spf", "SPF")
    data.columns = data.columns.str.replace("SPFh", "SPFH")

    return data
