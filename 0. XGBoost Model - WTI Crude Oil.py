'''
========== Model Coverage ==========
Symbols: CL
Product: Crude Oil WTI
Future Type: Energy
Data Source: https://firstratedata.com/b/29/futures-most-active
File Format: {Symbol}{Expire Month Code}{Expire Month Year}.txt | Example: CLF09.txt > {CL}{F = January}{09 = 2009}.txt
Coverage Period:
 > Pre 2021: All future contract starting from 2008-04 to 2020-12
 > 2021: All future contract from 2021-01 to 2021-12
 > Post 2022: No coverage
Data Granularity: 1 Minute Bars (OHLC with Volume); if volume was zero, the data point was dropped  




========== Model Purpose ==========
Purpose of the model:
    1) Identify anomalies in range
    2)
    3)
    4)
    5)




==========  Model Steps  ==========
1) Import and consolidate text files into one consolidated file  
2) Resample data from 1 minute bars into 20 minute bars and create metrics
3) Machine learning: Predict three periods into the future. 
4)
5)



==========  System Environment  ==========
OS Name: Microsoft Windows 10 Pro; Please note that if you are using an Linux or another OS, the Path package may require modification.
OS Version: 10.0.19043 Build 19043
Python Version: 3.9.2
Pandas: pandas==1.2.3
XGBoost: xgboost==1.3.3

'''

######################################## Required Packages ########################################
import pandas as pd
import glob
from pathlib import Path
from xgboost import XGBRegressor as XGBReg
import time as timetracker

masterstart = timetracker.time()
start = timetracker.time()
######################################## Model Step: 1 ######################################## 
''' Import and consolidate text files into one consolidated file '''
######################################## Model Step: 1 ######################################## 

# 1.1
'''
Comment: Import Symbols Map File
'''
symbols = pd.read_csv('1. Symbols.csv').dropna()
symbols = symbols[symbols['Symbol'] == 'CL'] ## The model has been designed to be compatible with any symbol, however, for demonstration purposes we will be only analyzing CL. To perform a full analysis, all that is necessary is to make sure that required contract data files for each symbol exist in contracts_pre_2021 & contracts_pre_2021.  
contractexpiry = pd.DataFrame({'Expiry Month Code': ['F','G','H','J','K','M','N','Q','U','V','X','Z'], 'Expiry Month': [1,2,3,4,5,6,7,8,9,10,11,12], 'Expiry Month Name': ['January','February','March','April','May','June','July','August','September','October','November','December']})

''' Comment: The settlement dates file (1. Contract Settlement DateTime (EST).csv) only contains settlement dates for CL. Note: if needed to expand to other symbols, make sure to update the file, below is the method used to generate the current file:
    > For contracts before 2021-07 (settled) - settlement date is assumed to be MAX(DateTime (EST))
    > For contract 2021-07 and later, settlement dates were sourced from: 'https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.calendar.html'
'''
settlementdates = pd.read_csv('1. Contract Settlement DateTime (EST).csv', parse_dates=['Contract Settlement DateTime (EST)'])

# 1.2
'''
Comment: Code below walks the following folders:
         > contracts_pre_2021, and
         > contracts_pre_2021.
         Each file found ending with .txt extension is saved into a list variable named 'files'
'''
files = []
for i in glob.glob('contracts_pre_2021/*.txt'):
    files.append(i)
for i in glob.glob('contracts_post_2021/*.txt'):
    files.append(i)

# 1.3
def importfile(filename):
    '''Comment: Function (importfile) is used to convert text files into a dataframe.
                The function then creates columns by extracting the contracts expire month & year from the filename. 
                The function that proceeds to save each dataframe into a consolidated csv named: '2. {Future Type} - {Product Name}.csv'
                Function (importfile) will be called inside another Function (compileconsolidated) compiled in the next step.
    '''
    global df
    filecolumns = ['DateTime (EST)', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(filename,parse_dates=[0])
    df.columns = filecolumns
    df['Symbol'] = str(filename[filename.find('\\')+1:filename.find('.')-3])
    df['Expiry Month Code'] = str(filename[filename.find('.')-3:filename.find('.')-2])
    df['Expiry Year'] = '20'+str(filename[filename.find('.')-2:filename.find('.')])
    df = df.merge(symbols, how='left', on='Symbol')
    df = df.merge(contractexpiry, how='left', on='Expiry Month Code')
    df['Contract'] = df['Expiry Year'].astype('str')+'-'+df['Expiry Month'].astype('str').str.zfill(2)
    df = df.merge(settlementdates, left_on=['Symbol','Contract'], right_on=['Symbol', 'Contract'])

    
    df_out = Path('2. {} - {}.csv'.format(symbols['Future Type'][symbols['Symbol'] == str(filename[filename.find('\\')+1:filename.find('.')-3])].item(),symbols['Product Name'][symbols['Symbol'] == str(filename[filename.find('\\')+1:filename.find('.')-3])].item()))
    if df_out.is_file():
        df.to_csv(df_out, mode='a', header=False,index=False)
    else:
        df.to_csv(df_out, mode='a', header=True,index=False)

# 1.4
def compileconsolidated(listoffiles):
    '''Comment: Function (compileconsolidated) takes a list of filenames as an input. It is used to loop through each file saved inside in the list.
                All files that result in an error/fail on calling the Function(importfile) are saved to a csv named:
                > '2. Import Errors.csv'. If '2. Import Errors.csv' is a blank file, it means no errors were detected.
    '''
    global t, errors
    t = 1
    errors = []
    for i in listoffiles:
        print('{} of {} - {}'.format(t, len(files), i))
        try:
            importfile(i)
        except:
            print('ERROR - {} of {} - {} - ERROR'.format(t, len(listoffiles), i))
            errors.append(i)
        t+=1
    if len(errors) == 0:
        print(' ')
        print('Files that resulted in errors: None')
        print(' ')
    else:
        print(' ')
        print('Files that resulted in errors: {}'.format(errors))
        print(' ')
    pd.DataFrame(errors).to_csv('2. Import Errors.csv',index=False)

# 1.5
''' Comment: The line below calls the function that was created in step 1.4 and passes the 'files' variable created in step 1.2 as the list of files to import and parse'''
compileconsolidated(files)

end = timetracker.time()
print("Elapsed time for consolidating text files into Master CSV: %s" % (end - start))



start = timetracker.time()
######################################## Model Step: 2 ######################################## 
''' Resample data from 1 minute bars into 20 minute bars '''
######################################## Model Step: 2 ######################################## 


# 2.1
def resampledata(df, timeframe):
    ''' Comment: This function leverages the pandas resample function. Link to pandas documentation (includes examples): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
                 Timeframe inputs (value can be changed accordingly): 1S = 1 Second, 1T = 1 Minute, 1H = 1 Hour, 1D = 1 Day, 1W = 1 Week | Link to potential timeframes: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects (However, requires adjustment to code below)
    '''
    contractinfo = df[['Contract', 'Symbol', 'Contract Settlement DateTime (EST)', 'Expiry Month Code', 'Expiry Year', 'Product Name', 'Future Type', 'Expiry Month', 'Expiry Month Name']].drop_duplicates()
    pricedata = df[['DateTime (EST)', 'Open', 'High', 'Low', 'Close', 'Volume','Contract']].copy()
    pricedata.sort_values(['Contract','DateTime (EST)'], ascending=True, inplace=True)
    pricedata = pricedata.set_index('DateTime (EST)')
    dfout =  pricedata.groupby(['Contract']).resample(timeframe).agg({'Open': 'first', 'High' : 'max', 'Low' : 'min', 'Close': 'last', 'Volume': 'sum'})
    dfout.reset_index(inplace=True)
    dfout['From'] = dfout['DateTime (EST)']
    dfout['To'] = dfout['From'] + pd.Timedelta(timeframe) - pd.Timedelta(seconds=1)
    dfout = dfout.merge(contractinfo, left_on='Contract', right_on='Contract')
    print('Original 1 minute bars dataframe had {} records. The resampled data has {} records. The resampled data timeframe is: {}'.format(format(len(df),'#,'),format(len(dfout),'#,'), timeframe))
    return dfout


# 2.2
def createmetrics(df):
    ''' Comment: Can be updated acccordingly.
    '''
    df['No Volume'] = df[['Open','High','Low','Close']].isna().sum(axis=1)
    check = list(set(df['No Volume']))
    check.sort()
    if check == [0,4]:
        pass
    else:
        print('This should not be possible, please review source data. The table below may provide some assistance.')
        print(df[~df['No Volume'].isin([0,4])])
        print('Type continue to "proceed" or "cancel" to stop:')
        choice = input()
        print('You have choosen to: {}'.format(choice))
        if choice == 'proceed':
            pass
        elif choice == 'cancel':
            print(ScriptStop)
        else:
            print('Incorrect response, script shutting down')
            print(ScriptStop)
            
    df.loc[df['No Volume'] == 4, 'No Volume'] = 1
    df.loc[df['No Volume'] == 0, 'No Volume'] = 0
    df['Close'] = df['Close'].fillna(method='ffill')
    df.loc[df['No Volume'] == 1, 'Open'] = df['Close']
    df.loc[df['No Volume'] == 1, 'High'] = df['Close']
    df.loc[df['No Volume'] == 1, 'Low'] = df['Close']
    df['Days Till Settlement'] = (df['Contract Settlement DateTime (EST)'] - df['To']) / pd.Timedelta(days=1)
    
    df['Close_Previous'] = df.groupby(['Future Type','Symbol','Contract'])['Close'].shift(1)
    df['DateTime (EST)_Previous'] = df.groupby(['Future Type','Symbol','Contract'])['DateTime (EST)'].shift(1)
    df['Minutes Since Last Trade'] = (df['DateTime (EST)'] - df['DateTime (EST)_Previous']) / pd.Timedelta(minutes=1)
    df['Open_Gap'] = df['Open'] - df['Close_Previous']
    df['Open_Gap%'] = df['Open_Gap'] / df['Close_Previous']
    df['Change'] = df['Close'] - df['Close_Previous']
    df['Change%'] = df['Change'] / df['Close_Previous']
    df['Range'] = (df['High'] - df['Low'])    
    df.loc[df['Close_Previous'] > df['High'], 'Range'] = (df['Close_Previous'] - df['Low'])
    df.loc[df['Close_Previous'] < df['Low'], 'Range'] = (df['High'] - df['Close_Previous'])
    df['Range%'] = df['Range'] / df['Close_Previous']
    df['DollarVolume'] = abs(df['Change'])*df['Volume']
    df.loc[df['High'] < df['Close_Previous'], 'DollarVolume_Positive'] = 0
    df.loc[df['High'] >= df['Close_Previous'], 'DollarVolume_Positive'] = (df['High']-df['Close_Previous'])*df['Volume']
    df.loc[df['Low'] <= df['Close_Previous'], 'DollarVolume_Negative'] = (df['Low']-df['Close_Previous'])*df['Volume']
    df.loc[df['Low'] > df['Close_Previous'], 'DollarVolume_Negative'] = 0
    df.loc[(df['DollarVolume_Positive'] != 0) & (df['DollarVolume_Positive'] != 0), 'GAP'] = 0
    df.loc[df['DollarVolume_Positive'] == 0, 'GAP'] = -1
    df.loc[df['DollarVolume_Negative'] == 0, 'GAP'] = 1
    df['Date_Date'] = df['DateTime (EST)'].dt.date
    df['Date_Year'] = df['DateTime (EST)'].dt.year
    df['Date_Month'] = df['DateTime (EST)'].dt.month
    df['Date_DayOfMonth'] = df['DateTime (EST)'].dt.day
    df['Date_DayOfWeek'] = df['DateTime (EST)'].dt.dayofweek+1
    df['Date_DayOfYear'] = df['DateTime (EST)'].dt.dayofyear
    df['Date_Quarter'] = df['DateTime (EST)'].dt.quarter
    df['Date_Week'] = df['DateTime (EST)'].dt.isocalendar().week 
    df['Date_Time'] = df['DateTime (EST)'].dt.time
    df['Date_Hour'] = df['DateTime (EST)'].dt.hour
    df['Date_Minute'] = df['DateTime (EST)'].dt.minute
    #df['Date_MonthName'] = df['DateTime (EST)'].dt.month_name()
    #df['Date_DayName'] = df['DateTime (EST)'].dt.day_name()   
    #df['Date_LeapYear'] = df['DateTime (EST)'].dt.is_leap_year
    #df['Date_DaysInMonth'] = df['DateTime (EST)'].dt.daysinmonth

    historicalmeancolumns = ['Close','Volume','DollarVolume','Open_Gap','Change','Range']

    for i in range(1,10):
        for column in historicalmeancolumns:
            df[column+'_ewm_'+str(i)] = df.groupby(['Future Type','Symbol','Contract'])[column].apply(lambda x: x.ewm(span=i, adjust=False).mean())

    for i in range(1,4):
        df['Close_Next_'+str(i)] = df.groupby(['Future Type','Symbol','Contract'])['Close'].shift(-i)
        df.loc[df['Close_Next_'+str(i)] > df['Close'], 'CloseBinary_Next_'+str(i)] = 1
        df.loc[df['Close_Next_'+str(i)] < df['Close'], 'CloseBinary_Next_'+str(i)] = -1
        df.loc[df['Close_Next_'+str(i)] == df['Close'], 'CloseBinary_Next_'+str(i)] = 0
        df['DateTime (EST)_Next_'+str(i)] = df.groupby(['Future Type','Symbol','Contract'])['DateTime (EST)'].shift(-i)

    


# 2.3
''' Comment: The list below is used as a placeholder to automate the looping through function(resampledata) created in step 2.1
             Acceptable inputs: 1S = 1 Second, 1T = 1 Minute, 1H = 1 Hour, 1D = 1 Day, 1W = 1 Week
'''
resampletimeframes = ['15T']

# 2.4
''' Comment: Create a list of Future Type products to iterate through'''
FutureType = list(set(symbols['Future Type']))
FutureType.sort()
print('Future Types detected: {}'.format(FutureType))


# 2.5
''' Comment: The code below is more complex than it needs to be, however, the benefit is immense as it creates named variable automatically and needed for automation.
             Key takeaway: Any item inside ProductNamedVariable or ProductNamedVariablesResampled can be accessed in two ways.
                  Example: In this example ProductNamedVariables contains ['Energy_CrudeOilWTI'], it can be accessed:
                           1. Refrencing Energy_CrudeOilWTI directly in the script or in the terminal as it is a named variable,
                           2. globals()[ProductNamedVariables[0]]                          
'''
             
ProductNamedVariables = [] ## if any items contains special characters, it can be accessed be using like globals()[key] as an example; if it were to contain 'S&P' with & being the special character, it can still be accessed using: globals()['S&P'].
ProductNamedVariablesResampled = [] ## if any items contains special characters, it can be accessed be using like globals()[key] as an example; if it were to contain 'S&P' with & being the special character, it can still be accessed using: globals()['S&P'].

for i in FutureType:
    FutureType = [i]
    FutureTypeProducts = list(set(symbols['Product Name'][symbols['Future Type'] == i]))
    FutureTypeProducts.sort()
    print('Future Types Products detected: {}'.format(FutureTypeProducts))
    print(' ')
        
    for p in FutureTypeProducts: 
        key = i+' - '+p
        value = pd.read_csv('2. '+key+'.csv', dtype={'Contract': 'object'}, parse_dates=['DateTime (EST)', 'Contract Settlement DateTime (EST)'])
        value.sort_values(by=['Future Type','Product Name','Contract','DateTime (EST)'], ascending=True, inplace=True)
        value.reset_index(drop=True)
        key_cleaned = key.replace(' - ','_').replace(' ','').replace('(','').replace(')','').replace('-','').replace('&','')
        globals()[key_cleaned] = value
        ProductNamedVariables.append(key_cleaned)

        for time in resampletimeframes:
            key_time = key_cleaned+'_'+time
            value = resampledata(globals()[key_cleaned], time)
            createmetrics(value)
            globals()[key_time] = value
            value.to_csv('3. '+i+' - '+p+'_'+time+' - Includes Metrics.csv', index=False)
            ProductNamedVariablesResampled.append(key_time)

print(' ')            
print('ProductNamedVariables: {}'.format(ProductNamedVariables))
print('ProductNamedVariablesResampled: {}'.format(ProductNamedVariablesResampled))
print(' ')

    
end = timetracker.time()
print("Elapsed time for resampling data timeframe and building metrics: %s" % (end - start))    
    
start = timetracker.time()
######################################## Model Step: 3 ######################################## 
''' Machine learning: Predict three periods into the future. '''
######################################## Model Step: 3 ######################################## 

def trainandpredict(df, splitpct, testgroups, target):
    global model, full, train, test, x, y, x_test, y_test
    narows = int(len(df[df.isna().any(axis=1)]))
    totalrows = int(len(df))
    print('{} rows contain NA values out of {} total rows. This is due to: 1) EWM results in the first row as blank, and 2) Since we are shifting the target columns three periods back to see the result of the prediction, it is therefore, unable to pull at the end of the dataset because the next three period dont exist. If it relatively very high compared to totalrows, that would mean one or more the columns have missing values. In that case, please validate data before continuing.'.format(format(narows,'#,'),format(totalrows,'#,')))
    full = df[~df.isna().any(axis=1)]
    full['RecordValue'] = 1
    full['CumulativeRecordValue'] = full.groupby(['Future Type','Symbol','Contract'])['RecordValue'].cumsum()
    totalrecords = full.groupby(['Future Type','Symbol','Contract'])['RecordValue'].sum().reset_index()
    totalrecords = totalrecords.rename(columns={'RecordValue':'TotalRecords'})
    full = full.merge(totalrecords, left_on=['Future Type','Symbol','Contract'], right_on=['Future Type','Symbol','Contract'])
    full['%ofRecord'] = full['CumulativeRecordValue'] / full['TotalRecords']

    columnstofactorize = ['Contract','DateTime (EST)', 'From', 'To', 'Contract Settlement DateTime (EST)', 'Date_Time', 'Date_Week', 'DateTime (EST)_Previous','Date_Date']
    inputcolumns = ['Days Till Settlement', 'Date_Hour','Date_Minute','Date_DayOfMonth', 'Date_DayOfWeek', 'Date_DayOfYear', 'Date_Quarter',
                    'Open', 'High', 'Low', 'Close', 'Volume',  'Close_Previous', 'Open_Gap', 'Open_Gap%', 'Change', 'Change%', 'Range', 'Range%', 
                    'No Volume', 'GAP', 'Minutes Since Last Trade', 'DollarVolume', 'DollarVolume_Positive', 'DollarVolume_Negative',
                    'Date_Year', 'Expiry Year', 'Expiry Month', 'Date_Month',                    
                    'Close_ewm_1', 'Volume_ewm_1', 'DollarVolume_ewm_1', 'Open_Gap_ewm_1', 'Change_ewm_1', 'Range_ewm_1',
                    'Close_ewm_2', 'Volume_ewm_2', 'DollarVolume_ewm_2', 'Open_Gap_ewm_2', 'Change_ewm_2', 'Range_ewm_2',
                    'Close_ewm_3', 'Volume_ewm_3', 'DollarVolume_ewm_3', 'Open_Gap_ewm_3', 'Change_ewm_3', 'Range_ewm_3',
                    'Close_ewm_4', 'Volume_ewm_4', 'DollarVolume_ewm_4', 'Open_Gap_ewm_4', 'Change_ewm_4', 'Range_ewm_4',
                    'Close_ewm_5', 'Volume_ewm_5', 'DollarVolume_ewm_5', 'Open_Gap_ewm_5', 'Change_ewm_5', 'Range_ewm_5',
                    'Close_ewm_6', 'Volume_ewm_6', 'DollarVolume_ewm_6', 'Open_Gap_ewm_6', 'Change_ewm_6', 'Range_ewm_6',
                    'Close_ewm_7', 'Volume_ewm_7', 'DollarVolume_ewm_7', 'Open_Gap_ewm_7', 'Change_ewm_7', 'Range_ewm_7',
                    'Close_ewm_8', 'Volume_ewm_8', 'DollarVolume_ewm_8', 'Open_Gap_ewm_8', 'Change_ewm_8', 'Range_ewm_8',
                    'Close_ewm_9', 'Volume_ewm_9', 'DollarVolume_ewm_9', 'Open_Gap_ewm_9', 'Change_ewm_9', 'Range_ewm_9']


    for i in columnstofactorize:
        full[i+'_Factorize'], unique = pd.factorize(full[i])
        inputcolumns.append(i+'_Factorize')


    datasetsplit = 1 - splitpct
    splitpctvalue = splitpct/testgroups


    testitems = {}
    for i in range(testgroups):
        key = 'x_'+str(i)
        key1 = 'y_'+str(i)
        value = full[inputcolumns][(full['%ofRecord'] > (datasetsplit+(i*splitpctvalue))) & (full['%ofRecord'] <= (datasetsplit+((i+1)*splitpctvalue)))]
        value1 = full[target][(full['%ofRecord'] > (datasetsplit+(i*splitpctvalue))) & (full['%ofRecord'] <= (datasetsplit+((i+1)*splitpctvalue)))]
        testitems[key,key1] = [value,value1]
        print('Testgroup {}: {} to {} based on %ofRecord column'.format(i, datasetsplit+(i*splitpctvalue), datasetsplit+((i+1)*splitpctvalue)))
    for n in testitems:
        name0 = n[0]
        val0 = testitems[n][0]
        name1 = n[1]
        val1 = testitems[n][1]
        globals()[name0] = val0
        globals()[name1] = val1

    valid = []
    for i in testitems.values():
        valid.append(i)

    train = full[full['%ofRecord'] < datasetsplit]
    test = full[full['%ofRecord'] >= datasetsplit]

    x = train[inputcolumns]
    y = train[target]
    x_test = test[inputcolumns]
    y_test = test[target]

    model = XGBReg
    params = {'n_estimators': 200
              ,'max_depth': 115
              ,'learning_rate': 0.01
              ,'verbosity': 1 
              ,'objective': 'reg:squarederror'
              ,'booster':'gbtree'
              ,'tree_method': 'auto' 
              ,'importance_type': 'gain'
              ,'gpu_id':0
              ,'subsample':0.350
              ,'colsample_bytree':0.350
              ,'min_child_weight':0.350
              ,'lambda':0.20
              ,'alpha':0.20
              }

    model = model(**params)
    model.fit(x,y,eval_metric='rmse', eval_set=valid)
    
    

    def printimportance(model, x):
        global features
        features = model.feature_importances_
        for i in range(len(x.columns)):
                print(str(x.columns[i])+': '+' '*(75-len(str(x.columns[i])))+str('{0:.{1}f}'.format(features[i]*100, 2))+'%')
        importance = pd.DataFrame()
        for i in range(len(x.columns)):
            feat = x.columns[i]
            value = features[i]
            record = [[feat,value]]
            importance = importance.append(record)
        importance.columns = ['Feature','Importance']
        importance.to_csv('4. '+ProductNamedVariablesResampled[Product]+' - Feature Importance.csv',index=False)

    printimportance(model, x)
    test[target+'_prediction'] = model.predict(x_test)
    train[target+'_prediction'] = model.predict(x)
    full.loc[full['%ofRecord'] < datasetsplit, 'RecordPartOf'] = 'Train'
    full.loc[full['%ofRecord'] >= datasetsplit, 'RecordPartOf'] = 'Test'
    full[target+'_prediction'] = model.predict(full[inputcolumns])
    test.to_csv('4. '+ProductNamedVariablesResampled[Product]+' - Test Prediction for '+target+' .csv',index=False)
    train.to_csv('4. '+ProductNamedVariablesResampled[Product]+' - Train Prediction for '+target+' .csv',index=False)
    full.to_csv('4. '+ProductNamedVariablesResampled[Product]+' - Combined Prediction for '+target+' .csv',index=False)


for Product in range(len(ProductNamedVariablesResampled)):
    trainandpredict(globals()[ProductNamedVariablesResampled[Product]], 0.20, 4, 'Close_Next_1')

end = timetracker.time()
print("Elapsed time for Train and Predict: %s" % (end - start))

masterend = timetracker.time()
print("Elapsed time from start to finish: %s" % (masterend - masterstart))

