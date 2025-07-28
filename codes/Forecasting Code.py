#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import concurrent.futures
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Remove: from pmdarima import auto_arima
# Add these:
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


# In[1]:




# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[4]:


input_file=r"C:\Users\Manasvi Kadhi\Desktop\EMCURE\Forecasting\IQVIA TSA All Pack Feb'25.xlsx"
df = pd.read_excel(input_file, sheet_name='DATA', index_col=0)


# In[5]:


df.head()


# In[6]:


X= df


# In[7]:


unit_list = [
    "UNIT MAR'20", "UNIT APR'20", "UNIT MAY'20", "UNIT JUN'20", "UNIT JUL'20", "UNIT AUG'20",
    "UNIT SEP'20", "UNIT OCT'20", "UNIT NOV'20", "UNIT DEC'20", "UNIT JAN'21", "UNIT FEB'21",
    "UNIT MAR'21", "UNIT APR'21", "UNIT MAY'21", "UNIT JUN'21", "UNIT JUL'21", "UNIT AUG'21",
    "UNIT SEP'21", "UNIT OCT'21", "UNIT NOV'21", "UNIT DEC'21", "UNIT JAN'22", "UNIT FEB'22",
    "UNIT MAR'22", "UNIT APR'22", "UNIT MAY'22", "UNIT JUN'22", "UNIT JUL'22", "UNIT AUG'22",
    "UNIT SEP'22", "UNIT OCT'22", "UNIT NOV'22", "UNIT DEC'22", "UNIT JAN'23", "UNIT FEB'23",
    "UNIT MAR'23", "UNIT APR'23", "UNIT MAY'23", "UNIT JUN'23", "UNIT JUL'23", "UNIT AUG'23",
    "UNIT SEP'23", "UNIT OCT'23", "UNIT NOV'23", "UNIT DEC'23", "UNIT JAN'24", "UNIT FEB'24",
    "UNIT MAR'24", "UNIT APR'24", "UNIT MAY'24", "UNIT JUN'24", "UNIT JUL'24", "UNIT AUG'24",
    "UNIT SEP'24", "UNIT OCT'24", "UNIT NOV'24", "UNIT DEC'24", "UNIT JAN'25", "UNIT FEB'25"
]
qty_list = [
    "QTY MAR'20", "QTY APR'20", "QTY MAY'20", "QTY JUN'20", "QTY JUL'20", "QTY AUG'20",
    "QTY SEP'20", "QTY OCT'20", "QTY NOV'20", "QTY DEC'20", "QTY JAN'21", "QTY FEB'21",
    "QTY MAR'21", "QTY APR'21", "QTY MAY'21", "QTY JUN'21", "QTY JUL'21", "QTY AUG'21",
    "QTY SEP'21", "QTY OCT'21", "QTY NOV'21", "QTY DEC'21", "QTY JAN'22", "QTY FEB'22",
    "QTY MAR'22", "QTY APR'22", "QTY MAY'22", "QTY JUN'22", "QTY JUL'22", "QTY AUG'22",
    "QTY SEP'22", "QTY OCT'22", "QTY NOV'22", "QTY DEC'22", "QTY JAN'23", "QTY FEB'23",
    "QTY MAR'23", "QTY APR'23", "QTY MAY'23", "QTY JUN'23", "QTY JUL'23", "QTY AUG'23",
    "QTY SEP'23", "QTY OCT'23", "QTY NOV'23", "QTY DEC'23", "QTY JAN'24", "QTY FEB'24",
    "QTY MAR'24", "QTY APR'24", "QTY MAY'24", "QTY JUN'24", "QTY JUL'24", "QTY AUG'24",
    "QTY SEP'24", "QTY OCT'24", "QTY NOV'24", "QTY DEC'24", "QTY JAN'25", "QTY FEB'25"
]
value_list = [
    "MAR'20", "APR'20", "MAY'20", "JUN'20", "JUL'20", "AUG'20",
    "SEP'20", "OCT'20", "NOV'20", "DEC'20", "JAN'21", "FEB'21",
    "MAR'21", "APR'21", "MAY'21", "JUN'21", "JUL'21", "AUG'21",
    "SEP'21", "OCT'21", "NOV'21", "DEC'21", "JAN'22", "FEB'22",
    "MAR'22", "APR'22", "MAY'22", "JUN'22", "JUL'22", "AUG'22",
    "SEP'22", "OCT'22", "NOV'22", "DEC'22", "JAN'23", "FEB'23",
    "MAR'23", "APR'23", "MAY'23", "JUN'23", "JUL'23", "AUG'23",
    "SEP'23", "OCT'23", "NOV'23", "DEC'23", "JAN'24", "FEB'24",
    "MAR'24", "APR'24", "MAY'24", "JUN'24", "JUL'24", "AUG'24",
    "SEP'24", "OCT'24", "NOV'24", "DEC'24", "JAN'25", "FEB'25"
]


# In[31]:


import pandas as pd

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == 'GYNAEC.'].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None

# Final output
gynac_forecast_units = gynaec_combined_wide

# In[148]:

sg="GYNAEC."
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
gynac_forecast_qty=gynaec_combined_wide

# In[47]:
sg="GYNAEC."
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
# In[148]:
gynac_forecast_value=gynaec_combined_wide







import pandas as pd
sg="ANTI DIABETIC"
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None

ANTI_DIABETIC_forecast_units=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_DIABETIC_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_DIABETIC_forecast_value=gynaec_combined_wide

sg="CARDIAC"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
CARDIAC_forecast_units=gynaec_combined_wide


# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
CARDIAC_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
CARDIAC_forecast_value=gynaec_combined_wide






sg="VITAMINS/MINERALS/NUTRIENTS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
vitamins_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
vitamins_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
vitamins_forecast_value=gynaec_combined_wide

sg="RESPIRATORY"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
RESPIRATORY_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
RESPIRATORY_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
RESPIRATORY_forecast_value=gynaec_combined_wide


sg="PAIN / ANALGESICS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
pain_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
pain_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
pain_forecast_value=gynaec_combined_wide


sg="GASTRO INTESTINAL"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
gastro_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
gastro_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
gastro_forecast_value=gynaec_combined_wide


sg="ANTI MALARIALS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_MALARIALS_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    if temp_df['y'].nunique() <= 1 or temp_df['y'].isnull().any():
        print(f"Skipping {subgroup} due to constant or invalid data.")
        continue

    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    forecast = sf.forecast(HORIZON, temp_df)
    forecast['SUBGROUP'] = subgroup
    all_forecasts.append(forecast)

# Step 6: Combine all forecasts
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_MALARIALS_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_MALARIALS_forecast_value=gynaec_combined_wide


sg="ANTI-INFECTIVES"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_INFECTIVES_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_INFECTIVES_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_INFECTIVES_forecast_value=gynaec_combined_wide


sg="ANTINEOPLAST/IMMUNOMODULATOR"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
IMMUNOMODULATOR_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
IMMUNOMODULATOR_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
IMMUNOMODULATOR_forecast_value=gynaec_combined_wide


sg="ANTI-PARASITIC"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_PARASITIC_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_PARASITIC_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_PARASITIC_forecast_value=gynaec_combined_wide


sg="ANTI-TB"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_TB_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_TB_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTI_TB_forecast_value=gynaec_combined_wide

sg="ANTIVIRAL"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTIVIRAL_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTIVIRAL_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
ANTIVIRAL_forecast_value=gynaec_combined_wide


sg="BLOOD RELATED"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
BLOOD_RELATED_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
BLOOD_RELATED_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
BLOOD_RELATED_forecast_value=gynaec_combined_wide


sg="DERMA"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
DERMA_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)



# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
DERMA_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
DERMA_forecast_value=gynaec_combined_wide


sg="HEPATOPROTECTIVES"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HEPATOPROTECTIVES_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HEPATOPROTECTIVES_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HEPATOPROTECTIVES_forecast_value=gynaec_combined_wide

sg="HORMONES"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HORMONES_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HORMONES_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
HORMONES_forecast_value=gynaec_combined_wide


sg="STOMATOLOGICALS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
STOMATOLOGICALS_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
STOMATOLOGICALS_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
STOMATOLOGICALS_forecast_value=gynaec_combined_wide


sg="NEURO / CNS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
neuro_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
neuro_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
neuro_forecast_value=gynaec_combined_wide


sg="OPHTHAL / OTOLOGICALS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTOLOGICALS_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTOLOGICALS_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTOLOGICALS_forecast_value=gynaec_combined_wide


sg="OTHERS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTHERS_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTHERS_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
OTHERS_forecast_value=gynaec_combined_wide


sg="PARENTERAL"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
PARENTERAL_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
PARENTERAL_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    if temp_df['y'].nunique() <= 1 or temp_df['y'].isnull().any():
        print(f"Skipping {subgroup} due to constant or invalid data.")
        continue

    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    forecast = sf.forecast(HORIZON, temp_df)
    forecast['SUBGROUP'] = subgroup
    all_forecasts.append(forecast)

# Step 6: Combine all forecasts
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
PARENTERAL_forecast_value=gynaec_combined_wide


sg="SEX STIMULANTS / REJUVENATORS"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
sex_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
sex_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
sex_forecast_value=gynaec_combined_wide


sg="UROLOGY"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
UROLOGY_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
UROLOGY_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
UROLOGY_forecast_value=gynaec_combined_wide


sg="VACCINES"
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + unit_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("UNIT ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' UNITS'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "UNITS" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' UNITS', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
VACCINES_forecast_units=gynaec_combined_wide

# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + qty_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("QTY ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' QTY'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "QTY" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' QTY', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
VACCINES_forecast_qty=gynaec_combined_wide
# Step 1: Aggregate
agg_dict = {col: 'sum' for col in unit_list + qty_list + value_list}
supergroup_df = X.groupby(['SUPERGROUP', 'SUBGROUP'], as_index=False).agg(agg_dict)

# Step 2: Melt units only
supergroup_df_units = supergroup_df[["SUPERGROUP", "SUBGROUP"] + value_list]
df_melted = supergroup_df_units.melt(id_vars=['SUPERGROUP', 'SUBGROUP'], var_name='Date', value_name='y')

# Step 3: Format Date
df_melted['Date'] = df_melted['Date'].str.replace("value ", "", regex=False)
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format="%b'%y")
df_melted['ds'] = df_melted['Date']  # KEEP AS datetime for StatsForecast

# Step 4: Filter original_df for GYNAEC.
original_df = df_melted[df_melted['SUPERGROUP'] == sg].copy()

# Forecasting parameters
# Forecasting parameters
SEASON_LENGTH = 12
FREQ = 'M'
HORIZON = 61
MIN_POINTS = 4  # Add this line to set minimum data points

# Step 5: Forecast loop
all_forecasts = []

for subgroup in original_df['SUBGROUP'].unique():
    temp_df = original_df[original_df['SUBGROUP'] == subgroup].copy()
    temp_df['unique_id'] = subgroup
    temp_df = temp_df[['unique_id', 'ds', 'y']].sort_values('ds')
    
    # ==== ADD THESE CHECKS RIGHT HERE ====
    if (len(temp_df) < MIN_POINTS or 
        temp_df['y'].nunique() <= 1 or 
        temp_df['y'].isnull().any() or 
        (temp_df['y'] == 0).all()):
        print(f"Skipping {subgroup} due to insufficient/invalid data")
        continue
    
    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON_LENGTH)],
        freq=FREQ,
        n_jobs=1
    )

    # ==== ADD ERROR HANDLING HERE ====
    try:
        forecast = sf.forecast(HORIZON, temp_df)
        forecast['SUBGROUP'] = subgroup
        all_forecasts.append(forecast)
    except ZeroDivisionError:
        print(f"Skipping {subgroup} due to ARIMA fitting error")
        continue
    except Exception as e:
        print(f"Skipping {subgroup} due to error: {str(e)}")
        continue

# Step 6: Combine all forecasts (rest of your code remains the same)
forecast_df = pd.concat(all_forecasts, ignore_index=True)


# Step 7: Format forecast output
gynaec_forecast_df = forecast_df[['SUBGROUP', 'ds', 'AutoARIMA']].copy()
gynaec_forecast_df.rename(columns={'ds': 'Date', 'AutoARIMA': 'Forecast'}, inplace=True)
gynaec_forecast_df['Month_Label'] = gynaec_forecast_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_forecast_wide = gynaec_forecast_df.pivot(index='SUBGROUP', columns='Month_Label', values='Forecast').reset_index()

# Step 8: Prepare historical wide format
gynaec_hist_df = original_df.copy()
gynaec_hist_df['Month_Label'] = gynaec_hist_df['Date'].dt.strftime("%b'%y") + ' value'

gynaec_hist_wide = gynaec_hist_df.pivot(index='SUBGROUP', columns='Month_Label', values='y').reset_index()

# Step 9: Combine historical + forecast
gynaec_combined_wide = pd.merge(
    gynaec_hist_wide, gynaec_forecast_wide, on='SUBGROUP', how='outer', suffixes=('', '_DROP')
)
gynaec_combined_wide = gynaec_combined_wide.loc[:, ~gynaec_combined_wide.columns.str.endswith('_DROP')]

# Step 10: Sort chronologically
date_cols = [col for col in gynaec_combined_wide.columns if col != 'SUBGROUP' and "value" in col]
sorted_dates = sorted(date_cols, key=lambda x: pd.to_datetime(x.replace(' value', ""), format="%b'%y"))
gynaec_combined_wide = gynaec_combined_wide[['SUBGROUP'] + sorted_dates]
gynaec_combined_wide.columns.name = None
VACCINES_forecast_value=gynaec_combined_wide
import pandas as pd

def combine_forecasts(supergroup, units_df, qty_df, value_df):
    # Add supergroup column
    units_df = units_df.copy()
    qty_df = qty_df.copy()
    value_df = value_df.copy()
    
    units_df['supergroup'] = supergroup
    qty_df['supergroup'] = supergroup
    value_df['supergroup'] = supergroup

    # Rename columns (except SUBGROUP and supergroup) to prevent overlap
    units_df = units_df.rename(columns=lambda x: f"{x} UNITS" if x not in ['SUBGROUP', 'supergroup'] else x)
    qty_df = qty_df.rename(columns=lambda x: f"{x} QTY" if x not in ['SUBGROUP', 'supergroup'] else x)
    value_df = value_df.rename(columns=lambda x: f"{x} VALUE" if x not in ['SUBGROUP', 'supergroup'] else x)

    # Merge on SUBGROUP and supergroup
    merged_df = pd.merge(units_df, qty_df, on=['SUBGROUP', 'supergroup'], how='outer')
    combined_df = pd.merge(merged_df, value_df, on=['SUBGROUP', 'supergroup'], how='outer')

    # Ensure column index name is cleared
    combined_df.columns.name = None

    # Reorder columns: keep 'supergroup', 'SUBGROUP' at front
    cols = ['supergroup', 'SUBGROUP'] + [col for col in combined_df.columns if col not in ['supergroup', 'SUBGROUP']]
    combined_df = combined_df[cols]

    return combined_df

# Combine all datasets for each supergroup
combined_dfs = []

datasets = [
    ('GYNAEC.', gynac_forecast_units, gynac_forecast_qty, gynac_forecast_value),
    ('ANTI DIABETIC', ANTI_DIABETIC_forecast_units, ANTI_DIABETIC_forecast_qty, ANTI_DIABETIC_forecast_value),
    ('VITAMINS/MINERALS/NUTRIENTS', vitamins_forecast_units, vitamins_forecast_qty, vitamins_forecast_value),
    ('CARDIAC', CARDIAC_forecast_units, CARDIAC_forecast_qty, CARDIAC_forecast_value),
    ('RESPIRATORY', RESPIRATORY_forecast_units, RESPIRATORY_forecast_qty, RESPIRATORY_forecast_value),
    ('PAIN / ANALGESICS', pain_forecast_units, pain_forecast_qty, pain_forecast_value),
    ('GASTRO INTESTINAL', gastro_forecast_units, gastro_forecast_qty, gastro_forecast_value),
    ('ANTI MALARIALS', ANTI_MALARIALS_forecast_units, ANTI_MALARIALS_forecast_qty, ANTI_MALARIALS_forecast_value),
    ('ANTI-INFECTIVES', ANTI_INFECTIVES_forecast_units, ANTI_INFECTIVES_forecast_qty, ANTI_INFECTIVES_forecast_value),
    ('ANTINEOPLAST/IMMUNOMODULATOR', IMMUNOMODULATOR_forecast_units, IMMUNOMODULATOR_forecast_qty, IMMUNOMODULATOR_forecast_value),
    ('ANTI-PARASITIC', ANTI_PARASITIC_forecast_units, ANTI_PARASITIC_forecast_qty, ANTI_PARASITIC_forecast_value),
    ('ANTI-TB', ANTI_TB_forecast_units, ANTI_TB_forecast_qty, ANTI_TB_forecast_value),
    ('ANTIVIRAL', ANTIVIRAL_forecast_units, ANTIVIRAL_forecast_qty, ANTIVIRAL_forecast_value),
    ('BLOOD RELATED', BLOOD_RELATED_forecast_units, BLOOD_RELATED_forecast_qty, BLOOD_RELATED_forecast_value),
    ('DERMA', DERMA_forecast_units, DERMA_forecast_qty, DERMA_forecast_value),
    ('HEPATOPROTECTIVES', HEPATOPROTECTIVES_forecast_units, HEPATOPROTECTIVES_forecast_qty, HEPATOPROTECTIVES_forecast_value),
    ('HORMONES', HORMONES_forecast_units, HORMONES_forecast_qty, HORMONES_forecast_value),
    ('NEURO / CNS', neuro_forecast_units, neuro_forecast_qty, neuro_forecast_value),
    ('OPHTHAL / OTOLOGICALS', OTOLOGICALS_forecast_units, OTOLOGICALS_forecast_qty, OTOLOGICALS_forecast_value),
    ('OTHERS', OTHERS_forecast_units, OTHERS_forecast_qty, OTHERS_forecast_value),
    ('PARENTERAL', PARENTERAL_forecast_units, PARENTERAL_forecast_qty, PARENTERAL_forecast_value),
    ('STOMATOLOGICALS', STOMATOLOGICALS_forecast_units, STOMATOLOGICALS_forecast_qty, STOMATOLOGICALS_forecast_value),
    ('SEX STIMULANTS / REJUVENATORS', sex_forecast_units, sex_forecast_qty, sex_forecast_value),
    ('UROLOGY', UROLOGY_forecast_units, UROLOGY_forecast_qty, UROLOGY_forecast_value),
    ('VACCINES', VACCINES_forecast_units, VACCINES_forecast_qty, VACCINES_forecast_value),
]

# Build final combined DataFrame
for supergroup, units_df, qty_df, value_df in datasets:
    try:
        combined_df = combine_forecasts(supergroup, units_df, qty_df, value_df)
        combined_dfs.append(combined_df)
    except Exception as e:
        print(f"❌ Error combining data for {supergroup}: {e}")

# Final concatenation
final_combined_df = pd.concat(combined_dfs, ignore_index=True)
import pandas as pd
import re

def clean_column_name(col):
    # Remove duplicated words like "UNITS UNITS" → "UNITS"
    col = re.sub(r"\b(\w+)\s+\1\b", r"\1", col)

    # Separate multi-part columns like "Mar'20 UNITS" into a clean format
    match = re.match(r"([A-Za-z']+\d{2})\s*(\w+)?", col)
    if match:
        date_part = match.group(1)
        type_part = match.group(2) if match.group(2) else ""
        return f"{date_part.strip()} {type_part.strip()}".strip()
    return col.strip()

# Apply the function to all columns
final_combined_df.columns = [clean_column_name(col) for col in final_combined_df.columns]
df= final_combined_df

# Extract the date columns for each category
unit_cols = [col for col in df.columns if "UNITS" in col]
qty_cols = [col for col in df.columns if "QTY" in col]
val_cols = [col for col in df.columns if "value" in col]
col=df.columns[13]
m=col[0:4]
# Helper function to generate MAT columns
def add_mat_cols(df, cols, label):
    mat_months = [col for col in cols if col.startswith(m)]
    for feb in mat_months:
        feb_idx = cols.index(feb)
        if feb_idx >= 11:
            mat_range = cols[feb_idx-11:feb_idx+1]  # Mar to Feb
            df[f"MAT {feb} {label}"] = df[mat_range].sum(axis=1)
    return df

# Apply to each category
df = add_mat_cols(df, unit_cols, "UNITS")
df = add_mat_cols(df, qty_cols, "QTY")
df = add_mat_cols(df, val_cols, "value")
# Fix column names with repeated words
df.columns = df.columns.str.replace(r"\b(UNITS|QTY|value)\s+\1\b", r"\1", regex=True)
summary_row = df.iloc[:, 2:].sum(numeric_only=True)  # Sum only numeric columns (excluding first two: 'supergroup', 'SUBGROUP')
summary_row['supergroup'] = 'IPM'
summary_row['SUBGROUP'] = 'total'

# Append to DataFrame:
df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)


output_file=r"C:\Users\Manasvi Kadhi\Desktop\EMCURE\Forecasting\forecasting_final.xlsx"
df.to_excel(output_file, index=False)