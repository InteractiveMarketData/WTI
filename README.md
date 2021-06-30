# WTI Crude Oil Forecasting Model | Built using XGBoost as the ML Algorithm | Python Build

==========  System Environment  ========== \
OS Name: Microsoft Windows 10 Pro; Please note that if you are using an Linux or another OS, the Path package may require modification. \
OS Version: 10.0.19043 Build 19043 \
Python Version: 3.9.2 \
Python Pandas: pandas==1.2.3 \
Python XGBoost: xgboost==1.3.3

========== Model Coverage ========== \
Symbols: CL \
Product Name: Crude Oil WTI \
Future Type: Energy  
Data Source: https://firstratedata.com/b/29/futures-most-active \
Data Granularity: 1 Minute Bars (OHLC with Volume); if volume was zero, the data point was dropped \
File Format: {Symbol}{Expire Month Code}{Expire Month Year}.txt | Example: CLF09.txt > {CL}{F = January}{09 = 2009}.txt \
Coverage Period:
 - Pre 2021: All future contract starting from 2008-04 to 2020-12 \
 - 2021: All future contract from 2021-01 to 2021-12 \
 - Post 2022: No coverage



