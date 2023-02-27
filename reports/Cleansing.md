
# Data Cleansing 
Data cleansing is the process of taking a “raw” dataset and making slight adjustments to ensure it is ready for analysis. Below is a list of the cleansing activity which was undertaken to prepare the data for analysis, additional detail and reasoning is provided in the following subsections.

- Timestamp alignment to exact 2-minute periods. 
- Cumulative meter data reversals. 
- Anomalous cumulative data removal – single point. 
- Anomalous cumulative data removal – from start of monitoring. 
- Relevelling data following a meter reset. 
- Incorrect column assignment for non-cumulative (temperature) data
- Removal of out-of-range temperatures. 
- Supplementary data cleansing – amending spelling or grammar variations. 
- Supplementary data cleansing – aligning property age ranges.

## Timestamp Alignment 
Each sensor and meter in the monitoring system sends readings at an average frequency of around 2-minutes. In the cleaned dataset, some variation in the period between each reading and two readings from the same sensor can be up to a maximum of 4-minutes apart. In addition, the timestamps are not synchronised between sensors, meaning that each sensor takes its readings at different times, independent of the other sensors. In the raw dataset, two consecutive readings could be any length of time apart, for example when we have gaps.

As a result of the above, to compare the readings from different sensors and perform analysis on the heat pump data, it is necessary to align the timestamps. The following process was followed to realign the timestamps for the cleansed dataset:

![Figure 6.1](figures/cleansing/Figure%206.1.png)
 
*Figure 6.1: Timestamp alignment process.*

As a result of the timestamp alignment, it is important to note that **the cleansed dataset may not always give the correct instantaneous readings.**

## Cumulative Meter Reversals 
Some monitoring equipment installation issues which can be seen within the raw dataset are the occasional installation of meters or sensors in the wrong orientation. The result of installing a cumulative meter in the wrong orientation is that the readings decrease over time. To check for this issue, daily differences in the cumulative meter readings are assessed. For situations where the daily differences are mostly decreasing, the readings are reversed within the cleansed dataset (for example, a reading of -1kWh is changed to 1kWh). This is demonstrated within Figure 6.2 and Figure 6.3. 
 
![Figure 6.2](figures/cleansing/Figure%206.2.png)

*Figure 6.2: A graph showing a reversed boiler heat meter resulting in consistent negative readings within the raw data.*
 
![Figure 6.3](figures/cleansing/Figure%206.3.png)

*Figure 6.3: A graph showing the reversed values within the cleaned dataset.*

## Anomalous Cumulative Data Removal 
### Anomalous Points within Cumulative Dataset
Another issue sometimes witnessed within the raw dataset is anomalous data points. These occur when a single datapoint is randomly much higher or lower than the surrounding datapoints. These are identified by having a value outside the range of: 
- 95% of the minimum of the 3 values prior to the point, and 
- 95% of the maximum of the 3 values after the point.

This differs from a meter reset (discussed in Section 6.2.4) as when a meter is reset or replaced, generally the meter reading will reduce significantly and then continue from the new start point along a similar trend to before.

To ensure ease of analysis, and eradicate the chance of false results, the single anomalous points are removed from the cleansed dataset. The method described above removes all single anomalous points from the cleansed dataset however, it does not account for and will not remove multi-point anomalies.

Multi-point anomalies occur when a series of datapoints is randomly much higher or lower than the surrounding data. Within the raw dataset, the multi-point anomalies which exist occur when the data readings reduce significantly for a short period of time before returning to the expected level. In this scenario, the reduction in data readings is read by the automated cleansing process as a meter reset, so initially the data is re-levelled as described in Section 6.2.4.

For multi-point anomalies, the data already returns to the expected level, so the re-levelling process may cause a sharp upward tick in the data. As a result of this, it is necessary to check the gradient of the cumulative data immediately after re-levelling. If the gradient is much greater than expected then a multi-point anomaly is assumed and the data is brought back in line with the previous point.

This single and multi-point anomaly removal process is demonstrated by Figure 6.4, Figure 6.5 and Figure 6.6. In Figure 6.4, four anomalies can be seen.

Figure 6.5 shows that, as anomalies 1, 2 and 4 are single point anomalies, they are removed however, anomaly 3 is a multi-point anomaly so is not removed. Instead, the figure shows that to resolve anomaly 3, the first anomalous point is re-levelled as described in Section 6.2.4. This then risks the data following the anomalous period to be erroneous (shown by the sharp increase in data readings). As there is a large gradient between two readings at the end of the anomalous period, the multi-point anomaly is identified and the last anomalous point is re-levelled to ensure it is aligned as expected.

Figure 6.6 shows the cleansed data with all anomalies resolved.  
 
![Figure 6.4](figures/cleansing/Figure%206.4.png)

*Figure 6.4: Four anomalies in a set of cumulative raw data.*

![Figure 6.5](figures/cleansing/Figure%206.5.png)
 
*Figure 6.5: Shows the single point anomaly removal for anomalies 1, 2 and 4 as well as an automated “meter reset” re-levelling for anomaly 3. (intermediate data cleansing stage)*
 
![Figure 6.6](figures/cleansing/Figure%206.6.png)

*Figure 6.6: Shows the re-levelling at the end of anomaly 3 due to a high gradient, thus resolving the multi-point anomaly in the cleansed data.*

### Anomalous Cumulative Data from Start of Monitoring 
As noted in Section 4.3.2, for some of the properties there were issues with the initial installation of the monitoring equipment resulting in either no reading being recorded or erroneous readings being recorded by the equipment. When these issues result in erroneous readings, they are generally represented by continuous anomalous data from the beginning of the monitoring period until the physical equipment issue is resolved. 
The result of this anomalous data is that, for a given duration at the beginning of the monitoring period, the Heat Pump Energy Output readings appear to track higher or lower than expected given the Whole System Energy Consumed. To find these periods, the Coefficient of Performance (COP) for each day was calculated (same calculation as SPFH2 but over the duration of a day rather than a year) and compared to the expected result. 
The data was rejected and removed if the daily COP was outside of the range 0.75-7.5. The data was only removed from the beginning of the monitoring period until the point where the daily COP falls within the range 0.75-7.5. 
This range is wider than the accepted range for annual SPF calculations or the COP values used for gap scoring. This is because a larger variation in heat pump efficiency is expected over a shorter timeframe. 
This issue is demonstrated by the graph shown in Figure 6.7. In the figure, the data is anomalous within the time period indicated by the grey box. To form the cleansed dataset, this data was removed and the meters relevelled to start at 0kWh, leaving only the non-anomalous data.

![Figure 6.7](figures/cleansing/Figure%206.7.png)
 
*Figure 6.7: Cumulative energy data from a given property whereby the data is anomalous from the start of the monitoring period, but then becomes aligned with expectation following this initial period.*

## Relevelling Data Following Meter Reset

A significant decrease in cumulative meter readings which does not return to the expected level is a likely result from a meter fault or meter replacement (where the readings immediately return to the expected level, this is an anomalous point, see Section 6.2.3.1). Small decreases in heat meter readings may be explained by the heat pumps running a defrost cycle, where the system draws energy from the home to defrost ice from the heat pump unit. A defrost cycle is part of the normal operation of a heat pump and it may be interesting to analyse the heat pump behaviour during these cycles. As such, it is necessary to differentiate between meter faults or replacements and defrost cycles when amending the data.

Meter faults and replacements often result in the meter being reset to (or near to) 0kWh. This reset usually occurs after a long gap in the data. A defrost cycle however will often result in a short, gradual decrease in the meter reading before the reading continues to increase along its previous trend. Meter resets have therefore been identified as decrease in the data where the reading drops by more than 95% of the previous reading and does not immediately return to the expected level.

If a meter reset is identified, the data is amended by relevelling all data following the meter reset such that the readings before and after the reset align. This means that the reading across the reset is flat, rather than increasing. If a gap exists prior to the reset, then energy usage across the gap is not considered. Instead, the gap is scored through the quality checks described in Section 6.3 and only a reset after a gap of less than 21 days of lost data may be included in the SPF analysis.

## Non-cumulative Data – Incorrect Column Assignment

For technical reasons relating to the monitoring system configurations (Section 4.14), the most likely data to be in the wrong columns are heat pump heating flow (HPHF), heat pump return (HPR) and hot water flow (HWF) temperatures.

This is because the HPHF and HWF temperatures are recorded using the same sensor, and the position of the control valve determines the direction of water flow and therefore which column the data should be recorded in. 

HPHF and HPR temperatures are recorded by different sensors however, these sensors can be attributed to the wrong column due to equipment installation issues or an issue with the transmission of the data. The flow and return temperatures tend to be very similar and the return temperature can regularly exceed the flow temperature when the heat pump is not operational. This makes the issue difficult to identify. In addition, this issue is very rare within the data. As a result of this, there have been no column reassignements made between HPHF and HPR temperatures.

### Heating Flow and Hot Water Flow Assignment

As the same sensor was used to measure HPHF and HWF temperature, it is sometimes the case that they are recorded in the same column. Alternatively, the data in these columns may be erroneously swapped (i.e. HPHF recorded in HWF column and vice versa). For some homes, these issues were fixed whilst monitoring was ongoing. The result of this is that, part way through the monitoring period, the data is correctly separated into the two columns or it is swapped so that the data are in the correct columns from the point of the fix onwards.

To evaluate whether the data was in the correct columns, data in each column was characterised using the following metrics: 
- mean: value, (mean of the values for one sensor)
- mean: difference, (mean of the differences between chronologically consecutive values for one sensor)
- standard deviation (std): value, (standard deviation of values for one sensor)
- standard deviation (std): difference, (standard deviation of the differences between chronologically consecutive values for one sensor)
- spikiness, (Root mean square difference of differences of the values of one sensor. A full definition of the function used can be found on [Github](https://github.com/ES-Catapult/spikiness))
- spikiness of the differences, (Root mean square difference of differences of the difference between values of one sensor.)
- mean: daily max, (mean of the daily maximum values for one sensor)
- mean: daily min, (mean of the daily minimum values for one sensor)
- mean: count per day. (mean of the daily number of readings for one sensor)

As there is generally a distinct difference between the nature of the recorded “Hot Water” temperatures and “Heating” temperatures, the data and associated metrics were labelled “Hot Water” (for HWF) and “Not Hot Water” for HPHF and HPR.

The metrics were then used to train two decision trees (one using all of the metrics except “mean: count per day” and the other using all of the metrics). These decision trees can be seen in Figure 6.8 and Figure 6.9. These trees were used to identify data which had different characteristics to the data with the same sensor label. For example, some sensors labelled as “Hot Water” were grouped by the tree as “Not Hot Water”). Where sensors were mis-grouped, this suggests that the sensor data are more similar to those of the other type and therefore the data may have been mislabelled. As a result of this, these sensors were flagged for review.

![Figure 6.8](figures/cleansing/Figure%206.8.png)

*Figure 6.8: Non-cumulative data grouping decision tree (excluding “mean: count per day”). Red rings have been used to highlight number of sensors which have been classified by the tree as different from how it is labelled.*
 
![Figure 6.9](figures/cleansing/Figure%206.9.png)

*Figure 6.9: Non-cumulative data grouping decision tree (including “mean: count per day”). Red rings have been used to highlight number of sensors which have been classified by the tree as different from how it is labelled.*

Most of the homes had 0 flagged sensors, indicating that the data was allocated to the correct column however, some properties were found to have a single flagged sensor and others had two. The homes were treated differently based on the number of flagged sensors.

The homes with two flagged columns were simpler to deal with as it was assumed that the data from these columns wholly assigned to the incorrect columns and so the data should be swapped. The swap was performed and sensors re-run through the decision tree to check that they had been correctly re-attributed. If these checks were passed, then the data was relabelled within the cleansed dataset.

Single flagged homes were assumed to be a case where one sensor was recording both HWF and HPHF and was then corrected by physically changing the monitoring setup within the property. For these instances, change point analysis was run on the “mean: count per day” of the data to detect the point where the physical change happened. If a change point was detected, then the data was split at that point, and allocated to the correct columns before and after the change. The sensors were then re-run through the decision trees to check that they had been correctly re-attributed. 

For these data, it was assumed that the data before the change was from both the HPHF and HWF. It is difficult to confidently differentiate which data was from each use case and this issue affects very few homes so all data before the change is retained in the HPHF temperature column.

A graph of the change point detection is shown in Figure 6.10 whereby the vertical line marks the detected change point plotted alongside the “mean: count per day” in blue. Note that whilst it is physically less likely, the swapped sensors at the change point could be HWF and HPR temperatures.

![Figure 6.10](figures/cleansing/Figure%206.10.png)
 
*Figure 6.10: A graph of the change point detection used to identify when physical changes to the monitoring system were made.*

## Removal of Out-of-Range Temperatures

The range of expected temperatures recorded by each sensor within the heat pump monitoring system is relatively predictable and therefore it is possible to spot anomalous values. To search for anomalous values, it is necessary to set acceptable ranges. Within the cleansing process, these temperature ranges were wide, to maximise the temperature data which can be used and avoid removing any correct values. The acceptable temperature ranges are presented in Table 6.1.

There are a small number of anomalous temperature values which are vastly different to the usual expected ranges. These anomalous values are removed from the cleansed dataset and therefore not included within the analysis. 

*Table 6.1: Accepted temperature ranges for each data column.*

| Data Column                        | Min Value ( ⷪC ) | Max Value ( ⷪC ) | Notes                                                                                                                                                                                           |
| ---------------------------------- | --------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Internal_Air_Temperature           | 0               | 40              | Based on Temperature Variations in UK Heated Homes Study [20] with a 5 ⷪC variation on either side.                                                                                              |
| External_Air_Temperature           | -27.2           | 40.3            | Based on record UK temperatures [21].                                                                                                                                                           |
| Hot_Water_Flow_Temperature         | 5               | 80              | Min value based on freezing temperature of water. Maximum value based on the highest temperature possible by the units installed as part of this study [22]. Both have an extra +5 ⷪC variation. |
| Heat_Pump_Return_Temperature       | 5               | 80              | See Hot_Water_Flow_Temperature                                                                                                                                                                  |
| Heat_Pump_Heating_Flow_Temperature | 5               | 80              | See Hot_Water_Flow_Temperature                                                                                                                                                                  |
| Brine_Flow_Temperature             | -10             | 30              | In the UK, GSHPs ground loop generally operate around 10 ⷪC all year around [23]. A 20 ⷪC variation has been allowed either side of this.                                                         |
| Brine_Return_Temperature           | -10             | 30              | See Brine_Flow_Temperature                                                                                                                                                                      |

