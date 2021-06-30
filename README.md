<h3 align="center"> WTI Crude Oil Forecasting Model | Built using XGBoost as the ML Algorithm | Python Build </h3>

**==========  System Environment  ==========** \
**OS Name:** Microsoft Windows 10 Pro; Please note that if you are using an Linux or another OS, the Path package may require modification. \
**OS Version:** 10.0.19043 Build 19043 \
**Python Version:** 3.9.2 \
**Python Pandas:** pandas==1.2.3 \
**Python XGBoost:** xgboost==1.3.3 \
**Total Runtime:** 
 - Consolidating text files into Master CSV: 0 minutes
 - Resampling data timeframes and building metrics: 0 minutes
 - Train on GPU: 0 minutes
 - Predict on GPU: 0 minutes   

**========== Model Coverage ==========** \
**Symbols:** CL \
**Product Name:** Crude Oil WTI \
**Future Type:** Energy  
**Data Source:** https://firstratedata.com/b/29/futures-most-active \
**Data Granularity:** 1 Minute Bars (OHLC with Volume); if volume was zero, the data point was dropped \
**File Format:** {Symbol}{Expire Month Code}{Expire Month Year}.txt | Example: CLF09.txt > {CL}{F = January}{09 = 2009}.txt \
**Coverage Period:**
 - Pre 2021: All future contract starting from 2008-04 to 2020-12
 - 2021: All future contract from 2021-01 to 2021-12
 - Post 2022: No coverage

**========== Model Summary ==========** \
**Data Timeframe:** 1 Minute Bars were resampled to 15 Minute Bars, zero volume timestamps have been added back and OHLC have been filled with last close \
**Data Time Period:** 2008-04 to 2021-12 WTI Crude Oil Future Contracts \
**Target:** Closing Price Three Periods Ahead \
**Input Data:**: OHLC with Volume, Range (H-L), Change, DollarVolume(Change * Volume), in addition, with their exponential moving average generated from the past 10 periods \
**Evaluation Metric:** RMSE | An 0.25 RMSE would mean that the models prediction is off by an absolute value of $0.25 when it tries to predict closing price for three periods ahead. This metric does not account for the direction of prediction. \
**Train Test Split:** For each Future Contract Month the oldest 80% of data was labelled as train with the most recent 20% as test data. The test data was further split into four groups of 5% each. \
**Result(RMSE)**: [0] = 0.37783, [1] = 0.55392, [2] = 0.99590, [3] = 1.69334 \
**Insight:** [0] is data that is the closest to the train data, there is a noticable trend that training on more recent data yields better performance.
**Feature Importance:**                   
