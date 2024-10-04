        
import pandas as pd
import numpy as np
import datetime as dt
import os as os
from .classes import Apm,Sum,PolarH10, DictionaryPlus
from io import BytesIO
from zipfile import ZipFile
from copy import deepcopy
from pandas.errors import EmptyDataError
import time

def in_list(origin, target):
    """
    origin = columns without suffix\n
    target = columns with suffix\n
    returns columns in target that starts with suffixes specified in origin
    """
    output = []
    for j in origin:
        for i in target:
            if i.startswith(j):
                output.append(i)
    return list(set(output))

def columns_no_counter(target):
    """
    """
    output = []
    for j in target:
        if j.endswith("_counter") == False:
            output.append(j)
    return output

def categorical_processing(
        df,
        variable,
        grouping=None,
        normalize=None,
        drop_duplicates="yes"):
    dict = {}
    if drop_duplicates == "yes":
        df = df.drop_duplicates(subset="household_id")
    if grouping is None:
        df = df.drop_duplicates(subset="household_id")
        data = df[variable].loc[df[variable].notna()].value_counts()
        for j in data.index:
            dict[j] = data[j]
    else:
        data = df.loc[df[grouping].notna() & df[variable].notna()]
        index = list(df[variable].value_counts().index)
        for y in data[grouping].value_counts().index:
            data_y = data[variable].loc[data[grouping] == y].value_counts()
            len_y = sum(data_y)
            dict_y = {}
            if normalize is not None:
                for j in index:
                    try:
                        dict_y[j] = data_y[j] / len_y
                    except BaseException:
                        dict_y[j] = 0
            else:
                for j in index:
                    try:
                        dict_y[j] = data_y[j]
                    except BaseException:
                        dict_y[j] = 0
            dict[y] = dict_y
    return dict

def reduce(file, resolution=1):
    """
    resolution can be 1 (for 1 second resolution) or greater (for greater resolution), the latter gives smaller file size. 
    """

    if resolution != 1:

        yy = [file.index[0]+i*pd.Timedelta("00:00:01") for i in range(
            0, int(((file.index[-1]-file.index[0]).total_seconds()))+1, resolution)]

        gg = file.drop(list(set(file.index).difference(set(yy))))

    return gg

def interpolate(file, original_resolution=300, resolution=1, gaps_delta=pd.Timedelta("00:06:00"), binary_columns=[], numeric_columns=[], add_binary_counter=True):
    """
    resolution can be 1 (for 1 second resolution) or greater (for greater resolution), the latter gives smaller file size. 
    """
    columns_to_add = columns_no_counter(list(file.columns))
    start = file.index[0]
    interval = pd.Timedelta("00:00:01")
    ss = file.index[-1]-file.index[0]
    hh = int((ss.total_seconds()+1))
    gg = pd.DataFrame(np.nan, columns=columns_to_add, index=[
                      start+i*interval for i in range(0, hh)])
    gg.index.rename("Datetime", inplace=True)
    gg = gg.loc[list(set(gg.index).difference(set(file.index)))]

    gg = pd.concat([gg,file])
    gg = gg.sort_index()


    for y in columns_to_add:
        if y in in_list(binary_columns,columns_to_add):
            gg[y] = gg[y].interpolate()

            t = list(gg[y])
            for i in range(len(t)):
                if np.isnan(t[i]):
                    t[i] = 0
                elif t[i] < 0.5:
                    t[i] = 0
                else:
                    t[i] = 1
            gg[y] = t

            if add_binary_counter:
                counter = []
                counter_n = 0
                if gg[y].iloc[0] == 1:
                    counter_n = counter_n + 1
                    counter.append(counter_n)
                else:
                    counter.append(np.nan)
                for k in range(1, len(gg)):
                    if (gg[y].iloc[k] == 1) & (gg[y].iloc[k-1] == 0):
                        counter_n = counter_n + 1
                        counter.append(counter_n)
                    elif (gg[y].iloc[k] == 1) & (gg[y].iloc[k-1] == 1):
                        counter.append(counter_n)
                    else:
                        counter.append(np.nan)
                gg[y+"_counter"] = counter

    for y in columns_to_add:
        if y in in_list(numeric_columns, columns_to_add):
            gg[y] = gg[y].infer_objects(copy=False).interpolate() if (gg[y].dtype == np.dtypes.ObjectDType) else gg[y].interpolate()

    gaps = []
    for j in range(len(file.index)-1):
        if file.index[j+1]-file.index[j] > gaps_delta:
            gaps.append([file.index[j], file.index[j+1]])

    gaps_list = []
    nr = int((original_resolution/resolution)/2)
    for i in gaps:
        ss = i[1]-i[0]
        hh = int(ss.total_seconds())+1
        for k in range(nr+1, hh-nr-1):
            gaps_list.append(i[0]+pd.Timedelta("00:00:01")*k)

    gg = gg.drop(gaps_list)

    if resolution != 1:
        yy = [gg.index[0]+i*pd.Timedelta("00:00:01") for i in range(
            0, int(((gg.index[-1]-gg.index[0]).total_seconds()))+1, resolution)]

        gg = gg.drop(list(set(gg.index).difference(set(yy))))

    return gg

def keep_interval(file,interval=None):
    """possible intervals:
    "5 seconds"
    "10 seconds"
    "30 seconds"
    "1 minute"
    "2 minutes"
    "5 minutes"
    "10 minutes"
    "30 minutes"
    or a custom tuple ([list of minutes],[list of seconds])
    """
    if interval==None:
        return file
    elif interval=="5 seconds":
        df = file.loc[[True if file.index[i].second in list(
            range(0, 60, 5)) else False for i in range(len(file))]]
    elif interval == "10 seconds":
        df = file.loc[[True if file.index[i].second in list(
            range(0, 60, 10)) else False for i in range(len(file))]]
    elif interval=="30 seconds":
        df = file.loc[[True if file.index[i].second in list(
            range(0, 60, 30)) else False for i in range(len(file))]]
    elif interval=="1 minute":
        df = file.loc[[True if file.index[i].second == 0 else False for i in range(len(file))]]
    elif interval == "2 minutes":
        df = file.loc[[True if (file.index[i].second == 0) & (
            file.index[i].minute in list(range(0, 60, 2))) else False for i in range(len(file))]]
    elif interval == "5 minutes":
        df = file.loc[[True if (file.index[i].second == 0) & (
            file.index[i].minute in list(range(0,60,5))) else False for i in range(len(file))]]
    elif interval == "10 minutes":
        df = file.loc[[True if (file.index[i].second == 0) & (
            file.index[i].minute in list(range(0, 60, 10))) else False for i in range(len(file))]]
    elif interval == "30 minutes":
        df = file.loc[[True if (file.index[i].second == 0) & (
            file.index[i].minute in list(range(0, 60, 30))) else False for i in range(len(file))]]
    else:
        df = file.loc[[True if (file.index[i].second in interval[1]) & (
            file.index[i].minute in interval[0]) else False for i in range(len(file))]]
    return df

def add_binary_counter(file, gaps_delta=pd.Timedelta("00:05:00"), binary_columns=["cooking"]):
    """
    """
    gg = pd.DataFrame()
    gg["Datetime"] = file.index
    gg = gg.set_index("Datetime")

    for k in file.columns:
        s = []
        for j in gg.index:
            if j in file.index:
                s.append(file[k].loc[j])
            else:
                s.append(None)
        gg[k] = s

    for y in binary_columns:

        counter = []
        counter_n = 0
        if gg[y].iloc[0] == 1:
            counter_n = counter_n + 1
            counter.append(counter_n)
        else:
            counter.append(np.nan)
        for k in range(1, len(gg)):
            if (gg[y].iloc[k] == 1) & (gg[y].iloc[k-1] == 0):
                counter_n = counter_n + 1
                counter.append(counter_n)
            elif (gg[y].iloc[k] == 1) & (gg[y].iloc[k-1] == 1) & ((gg.index[k]-gg.index[k-1])>gaps_delta*2):
                counter_n = counter_n + 1                
                counter.append(counter_n)
            elif (gg[y].iloc[k] == 1) & (gg[y].iloc[k-1] == 1) & ((gg.index[k]-gg.index[k-1]) <= gaps_delta*2):
                counter.append(counter_n)
            else:
                counter.append(np.nan)
        gg[y+"_counter"] = counter

    return gg

def add_combined_counter(file, gaps_delta=pd.Timedelta("00:05:00"), binary_columns=["cooking"]):
    """
    """

    counter = []

    for y in binary_columns:

        set_columns = in_list([y], file.columns)

        counter = []
        counter_n = 0
        if file[set_columns].iloc[0].sum() != 0:
            counter_n = counter_n + 1
            counter.append(counter_n)
        else:
            counter.append(np.nan)
        for k in range(1, len(file)):
            if (file[set_columns].iloc[k].sum() != 0) & (file[set_columns].iloc[k-1].sum() == 0):
                counter_n = counter_n + 1
                counter.append(counter_n)
            elif (file[set_columns].iloc[k].sum() != 0) & (file[set_columns].iloc[k-1].sum() != 0) & ((file.index[k]-file.index[k-1]) > gaps_delta*2):
                counter_n = counter_n + 1
                counter.append(counter_n)
            elif (file[set_columns].iloc[k].sum() != 0) & (file[set_columns].iloc[k-1].sum() != 0) & ((file.index[k]-file.index[k-1]) <= gaps_delta*2):
                counter.append(counter_n)
            else:
                counter.append(np.nan)
        file[y+"_counter"] = counter

    return file

def sum_merge(files: DictionaryPlus):
    if len(files) == 1:
        return files.show()

    else:
        file = add_combined_counter(files.show().drop(columns=["cooking_counter"]).join(
            [files.show(i).drop(columns=["cooking_counter"]) for i in range(1, len(files))], sort=True, how="outer"))

        meta_keys = files.metadata()
        for k in meta_keys:
            a = []
            for j in files.values():
                if type(j.meta[k]) == type([]):
                    for z in j.meta[k]:
                        a.append(z)
                else:
                    a.append(j.meta[k])
            file.meta[k] = list(set(a))

        return file

def gen_merge(files):
    if len(files) == 1:
        file = files[0]
    else:
        file = files[0].join([i for i in files[1:]], sort=True, how="inner")

    return file

##############

def blank_filter(df, variables):
    """Filters a dataframe of blank values for all the columns
    included in variables
    """
    for i in variables:
        df = df.loc[~df[i].isna()]
    return df

##############

def to_timedelta(x):
    hours, minutes, seconds = int(x.split(":")[0]), int(
        x.split(":")[1]), int(x.split(":")[2])
    return dt.timedelta(0, seconds, 0, 0, minutes, hours)

def to_datetime(x):
    year, month, day = int(x.split('T')[0].split(
        '/')[0]), int(x.split('T')[0].split('/')[1]), int(x.split('T')[0].split('/')[2])
    hour, minute, second = int(x.split('T')[1].split(
        ':')[0]), int(x.split('T')[1].split(':')[1]), int(x.split('T')[1].split(':')[2][:-1])
    return dt.datetime(year, month, day, hour, minute, second)

def to_datetime_polar(x):
    year, month, day = int(x.split('T')[0].split(
        '-')[0]), int(x.split('T')[0].split('-')[1]), int(x.split('T')[0].split('-')[2])
    hour, minute, second, microsecond = int(x.split('T')[1].split(
        ':')[0]), int(x.split('T')[1].split(':')[1]), int(x.split('T')[1].split(':')[2].split(".")[0]), int(x.split('T')[1].split(':')[2].split(".")[1])*1000
    return dt.datetime(year, month, day, hour, minute, second, microsecond)

def hrv_comma_check(x):
    if type(x) == type(""):
        return float(x.replace(",","."))
    else:
        return float(x)

def remove_odd_characters(x):
    if type(x) == type(''):
        try:
            return float(x.split('_')[0])
        except:
            return np.nan
    else:
        return x

def upas_processing(directory, file):
    numeric = ['PumpingFlowRate',
               'OverallFlowRate',
               'SampledVolume',
               'FilterDP',
               'BatteryCharge',
               'AtmoT',
               'AtmoP',
               'AtmoRH',
               'AtmoDensity',
               'AtmoAlt',
               'GPSQual',
               'GPSlat',
               'GPSlon',
               'GPSalt',
               'GPSsat',
               'GPSspeed',
               'GPShDOP',
               'AccelX',
               'AccelXVar',
               'AccelXMin',
               'AccelXMax',
               'AccelY',
               'AccelYVar',
               'AccelYMin',
               'AccelYMax',
               'AccelZ',
               'AccelZVar',
               'AccelZMin',
               'AccelZMax',
               'RotX',
               'RotXVar',
               'RotXMin',
               'RotXMax',
               'RotY',
               'RotYVar',
               'RotYMin',
               'RotYMax',
               'RotZ',
               'RotZVar',
               'RotZMin',
               'RotZMax',
               'Xup',
               'XDown',
               'Yup',
               'Ydown',
               'Zup',
               'Zdown',
               'StepCount',
               'LUX',
               'UVindex',
               'HighVisRaw',
               'LowVisRaw',
               'IRRaw',
               'UVRaw',
               'PMMeasCnt',
               'PM1MC',
               'PM1MCVar',
               'PM2_5MC',
               'PM2_5MCVar',
               'PM4MC',
               'PM4MCVar',
               'PM10MC',
               'PM10MCVar',
               'PM0_5NC',
               'PM0_5NCVar',
               'PM1NC',
               'PM1NCVar',
               'PM2_5NC',
               'PM2_5NCVar',
               'PM4NC',
               'PM4NCVar',
               'PM10NC',
               'PM10NCVar',
               'PMtypicalParticleSize',
               'PMtypicalParticleSizeVar',
               'PM2_5SampledMass',
               'PMReadingErrorCnt',
               'PMFanErrorCnt',
               'PMLaserErrorCnt',
               'PMFanSpeedWarn',
               'PCB1T',
               'PCB2T',
               'FdpT',
               'AccelT',
               'PT100R',
               'PCB2P',
               'PumpPow1',
               'PumpPow2',
               'PumpV',
               'MassFlow',
               'MFSVout',
               'BFGenergy',
               'BattVolt',
               'v3_3',
               'v5',
               'PumpsON',
               'Dead',
               'BCS1',
               'BCS2',
               'BC_NPG',
               'FLOWCTL',
               'GPSRT',
               'SD_DATAW',
               'SD_HEADW',
               'TPumpsOFF',
               'TPumpsON',
               'CO2',
               'SCDT',
               'SCDRH',
               'VOCRaw',
               'NOXRaw']
    dtformat = '%Y-%m-%dT%H:%M:%S'
    df = pd.read_csv(directory+file, skiprows=list(range(114)) +
                     [115], index_col="DateTimeLocal", date_format={'DateTimeLocal': dtformat, 'DateTimeUTC': dtformat})
    df1 = open(directory+file).readlines()[0:107]
    PMSensorInterval = int(df1[44].split(',')[1])
    LogInterval = int(df1[47].split(',')[1])
    PowerSaveMode = int(df1[50].split(',')[1])
    df["SampleTime"] = df["SampleTime"].map(to_timedelta)
    df = interpolate(df, LogInterval, 1, pd.Timedelta(seconds=LogInterval*2), numeric_columns=numeric, add_binary_counter=False)
    if LogInterval==1:
        pass
    elif (LogInterval <= 5) & (LogInterval > 1):
        df = keep_interval(df, '5 seconds')
    elif (LogInterval <= 10) & (LogInterval > 5):
        df = keep_interval(df, '10 seconds')
    elif (LogInterval <= 30) & (LogInterval > 10):
        df = keep_interval(df, '30 seconds')
    elif (LogInterval <= 60) & (LogInterval > 30):
        df = keep_interval(df, '1 minute')
    elif (LogInterval <= 120) & (LogInterval > 60):
        df = keep_interval(df, '2 minutes')
    elif (LogInterval <= 300) & (LogInterval > 120):
        df = keep_interval(df, '5 minutes')
    elif (LogInterval <= 600) & (LogInterval > 300):
        df = keep_interval(df, '10 minutes')
    elif (LogInterval <= 3600) & (LogInterval > 600):
        df = keep_interval(df, '30 minutes')
    else:
        pass
    pmSensorColumns = ["PMMeasCnt",
                       "PM1MC",
                       "PM1MCVar",
                       "PM2_5MC",
                       "PM2_5MCVar",
                       "PM4MC",
                       "PM4MCVar",
                       "PM10MC",
                       "PM10MCVar",
                       "PM0_5NC",
                       "PM0_5NCVar",
                       "PM1NC",
                       "PM1NCVar",
                       "PM2_5NC",
                       "PM2_5NCVar",
                       "PM4NC",
                       "PM4NCVar",
                       "PM10NC",
                       "PM10NCVar",
                       "PMtypicalParticleSize",
                       "PMtypicalParticleSizeVar",
                       "PM2_5SampledMass",
                       "PMReadingErrorCnt",
                       "PMFanErrorCnt",
                       "PMLaserErrorCnt",
                       "PMFanSpeedWarn"]
    for j in pmSensorColumns:
        if PMSensorInterval == 0:
            df[j]=np.nan
        elif PMSensorInterval == 1:
            pass
        elif (PMSensorInterval >= 2) & (PMSensorInterval <= 15):
            df[j] = [df[j].iloc[i] if ((df.index[i].minute % PMSensorInterval == 0) & (
                df.index[i].second == 0)) else np.nan for i in range(len(df))]
        elif (PMSensorInterval ==16):
            df[j] = [df[j].iloc[i] if (
                (df.index[i].second % 30 == 0)) else np.nan for i in range(len(df))]
        elif (PMSensorInterval == 17) | (PMSensorInterval == 18):
            df[j] = [df[j].iloc[i] if (
                (df.index[i].second == 0)) else np.nan for i in range(len(df))]
    if PowerSaveMode == 1:
        for j in ["GPSQual",
                  "GPSlat",
                  "GPSlon",
                  "GPSalt",
                  "GPSsat",
                  "GPSspeed",
                  "GPShDOP"]:
            df[j] = [df[j].iloc[i] if ((df.index[i].hour <21) & (
                df.index[i].hour >= 4)) else np.nan for i in range(len(df))]
        for j in pmSensorColumns:
            df[j] = [np.nan if (((df.index[i].minute % 15 != 0) | (
                df.index[i].second != 0)) & ((df.index[i].hour >= 21) | (
                df.index[i].hour < 4))) else df[j].iloc[i] for i in range(len(df))]
            

    out = Apm(df)
    
    out.meta['header'] = df1
    out.meta['upasid'] = df1[2].split(',')[1]
    out.meta['samplename'] = df1[26].split(',')[1].strip('_')
    out.meta['cartridgeid'] = df1[27].split(',')[1].strip('_')

    return out

def pur_average(pur: pd.DataFrame):
    a = 0.524
    b = -0.0862
    c = 5.75
    rh = 'current_humidity'
    ch1 = 'pm2_5_cf_1'
    ch2 = 'pm2_5_cf_1_b'
    chmean = (pur[ch2]+pur[ch1])/2
    chmagn = abs(pur[ch2]-pur[ch1])
    chper = chmagn/chmean
    chclean = pd.Series([chmean.iloc[i] if ((chper.iloc[i] < 0.25) | (
        chmagn.iloc[i] < 15)) else np.nan for i in range(len(chmean))], index=pur.index)
    chadj = (a*chclean)+(b*pur[rh])+c
    pur['pm_adj'] = chadj

def purple_processing(directory, interpolation=1, interval="30 seconds", timezone_shift = dt.timedelta(hours=0) ):
    numeric = ['current_temp_f',
               'current_humidity',
               'current_dewpoint_f',
               'pressure',
               'mem',
               'rssi',
               'uptime',
               'pm1_0_cf_1',
               'pm2_5_cf_1',
               'pm10_0_cf_1',
               'pm1_0_atm',
               'pm2_5_atm',
               'pm10_0_atm',
               'pm2.5_aqi_cf_1',
               'pm2.5_aqi_atm',
               'p_0_3_um',
               'p_0_5_um',
               'p_1_0_um',
               'p_2_5_um',
               'p_5_0_um',
               'p_10_0_um',
               'pm1_0_cf_1_b',
               'pm2_5_cf_1_b',
               'pm10_0_cf_1_b',
               'pm1_0_atm_b',
               'pm2_5_atm_b',
               'pm10_0_atm_b',
               'pm2.5_aqi_cf_1_b',
               'pm2.5_aqi_atm_b',
               'p_0_3_um_b',
               'p_0_5_um_b',
               'p_1_0_um_b',
               'p_2_5_um_b',
               'p_5_0_um_b',
               'p_10_0_um_b']
    files = [i for i in os.listdir(directory) if i.split(".")[-1] == "csv"]
    if len(files)==0:
        print("there are no csv files in directory "+directory)
        return
    files = [pd.read_csv(directory+i) for i in files]
    df = pd.concat(files)
    if len(df["mac_address"].value_counts()) != 1:
        print("there is more than one or less than one mac address on directory "+directory)
        return
    df['UTCDateTime'] = df['UTCDateTime'].map(to_datetime)
    df.set_index('UTCDateTime', inplace=True)
    if df.index.value_counts().max() != 1:
        print('index duplicates in directory '+ directory)
        return
    df.sort_index(inplace=True)
    df[numeric] = df[numeric].map(remove_odd_characters)
    df = interpolate(df, 120, interpolation, pd.Timedelta(
        '00:04:00'), numeric_columns=numeric, add_binary_counter=False)
    df = keep_interval(df, interval)
    df.index = df.index + timezone_shift
    pur_average(df)
    return Apm(df)

def lascar_processing(directory, file, interpolation=1,interval="30 seconds"):
    numeric = ['CO(ppm)']
    dtformat = '%Y-%m-%d %H:%M:%S'
    df = pd.read_csv(directory+file,  index_col="Time",
                     date_format={'Time': dtformat}, usecols=['Time', 'CO(ppm)'])
    df = interpolate(df, 30, interpolation, pd.Timedelta(
        '00:01:00'), numeric_columns=numeric, add_binary_counter=False)
    df = keep_interval(df, interval)
    return Apm(df)

def sum_interpolation(file, interpolation=1, interval="5 minutes", timing=False):
    if timing:
        start = time.process_time()
    numeric = ['dot_temperature']
    binary = ['cooking']
    df = interpolate(file, 300, interpolation, pd.Timedelta(
        '00:06:00'), numeric_columns=numeric, binary_columns=binary, add_binary_counter=True)
    df = keep_interval(df, interval)
    if type(file) == type(Sum()):
        df=Sum(df)
        df.meta = file.meta
        if timing:
            end = time.process_time()
            print(f"{end-start} seconds")
        return df            
    else:
        if timing:
            end = time.process_time()
            print(f"{end-start} seconds")
        return Sum(df)

def sum_processing(zipname,processor_name = [],return_data=False,return_csv=True):

    def to_datetime_metrics(x):
        year, month, day = int(x.split('T')[0].split(
            '-')[0]), int(x.split('T')[0].split('-')[1]), int(x.split('T')[0].split('-')[2])
        hour, minute, second = int(x.split('T')[1].split(
            ':')[0]), int(x.split('T')[1].split(':')[1]), int(x.split('T')[1].split(':')[2].split("+")[0])
        return dt.datetime(year, month, day, hour, minute, second)


    def to_datetime_events(x):
        year, month, day = int(x.split('T')[0].split(
            '-')[0]), int(x.split('T')[0].split('-')[1]), int(x.split('T')[0].split('-')[2])
        hour, minute, second = int(x.split('T')[1].split(
            ':')[0]), int(x.split('T')[1].split(':')[1]), int(x.split('T')[1].split(':')[2].split("Z")[0])
        return dt.datetime(year, month, day, hour, minute, second)

    archive = ZipFile(zipname)

    #mission_logs = pd.read_csv(BytesIO(archive.read('mission_logs.csv')))
    events = pd.read_csv(BytesIO(archive.read('events.csv')))
    events["start_time"] = events["start_time"].map(to_datetime_events)
    events["stop_time"] = events["stop_time"].map(to_datetime_events)

    #sensors = pd.read_csv(BytesIO(archive.read('sensor.csv')))
    tags = pd.read_csv(BytesIO(archive.read('tags.csv')))
    #missions = pd.read_csv(BytesIO(archive.read('missions.csv')))
    metrics = DictionaryPlus()

    for i in archive.namelist():
        if ("metrics/" in i) & (len(i) > 8):
            name = i.split('/')[1]
            try:
                metrics[name] = pd.read_csv(
                    BytesIO(archive.read(i)), index_col="timestamp")
                metrics[name].index = metrics[name].index.map(to_datetime_metrics)
                metrics[name].drop(
                    axis=1, labels=['channel', 'sensor_type_id', "created_at"], inplace=True)
                metrics[name].rename(
                    columns={'value': 'dot_temperature'}, inplace=True)
                metrics[name] = Sum(metrics[name])
                metrics[name].meta["mission_id"] = "-".join(name.split(".")[0].split(
                    "-")[-5:-1])+"-"+(name.split(".")[0].split("-")[-1].upper())
                metrics[name].meta["meter_name"] = "-".join(name.split(".")[0].split(
                    "-")[0:2])
                metrics[name].meta['tags'] = list(tags['tag'].loc[tags['mission_id']==metrics[name].meta['mission_id']])
            except EmptyDataError:
                print(f"empyt metric {i}")

    for key, value in metrics.items():
        value['cooking'] = 0
        for j in range(len(events)):
            if (events['mission_id'].iloc[j] == value.meta["mission_id"]) and (events['processor_name'].iloc[j] in processor_name):
                for k in range(len(value)):
                    if (value.index[k] >= events['start_time'].iloc[j]) & (value.index[k] < events['stop_time'].iloc[j]):
                        value.loc[value.index[k],'cooking'] = 1
    
    if return_csv:
        for key,value in metrics.items():
            value.to_csv(key)
    
    if return_data:
        for key, value in metrics.items():
            out = deepcopy(value)
            out = Sum(add_binary_counter(out))
            out.meta = value.meta
            metrics[key] = out
        return metrics

def polar_processing(directory):

    files = [i.split(".")[0] for i in os.listdir(directory+"/") if (i.split(".")
                                                                    [-1] == "txt") & (i.split(".")[0].split("_")[0] == "Polar")]
    if len(files) == 0:
        print("there are no data files in directory"+directory)
        return
    if len(set([i.split("_")[2] for i in files])) > 1:
        print("the data files in this directory are from different sensors")
        return
    sensorID = [i.split("_")[2] for i in files][0]

    out = PolarH10()
    out.meta["sensorID"] = sensorID

    data = [pd.read_csv(directory+"/"+i+".txt", delimiter=";")
            for i in files if i.split("_")[-1] == "ECG"]
    if len(data) > 0:
        df = pd.concat(data)
        df['Phone timestamp'] = df['Phone timestamp'].map(to_datetime_polar)
        df.set_index('Phone timestamp', inplace=True)
        df.sort_index(inplace=True)
        ecg = Apm(df)
        ecg.meta["sensorID"] = sensorID
        out["ecg"] = ecg
    data = [pd.read_csv(directory+"/"+i+".txt", delimiter=";")
            for i in files if i.split("_")[-1] == "ACC"]
    if len(data) > 0:
        df = pd.concat(data)
        df['Phone timestamp'] = df['Phone timestamp'].map(to_datetime_polar)
        df.set_index('Phone timestamp', inplace=True)
        df.sort_index(inplace=True)
        acc = Apm(df)
        acc.meta["sensorID"] = sensorID
        out["acc"] = acc
    data = [pd.read_csv(directory+"/"+i+".txt", delimiter=";")
            for i in files if i.split("_")[-1] == "RR"]
    if len(data) > 0:
        df = pd.concat(data)
        df['Phone timestamp'] = df['Phone timestamp'].map(to_datetime_polar)
        df.set_index('Phone timestamp', inplace=True)
        df.sort_index(inplace=True)
        rr = Apm(df)
        rr.meta["sensorID"] = sensorID
        out["rr"] = rr
    data = [pd.read_csv(directory+"/"+i+".txt", delimiter=";")
            for i in files if i.split("_")[-1] == "HR"]
    if len(data) > 0:
        df = pd.concat(data)
        df['Phone timestamp'] = df['Phone timestamp'].map(to_datetime_polar)
        df.set_index('Phone timestamp', inplace=True)
        df.sort_index(inplace=True)
        hr = Apm(df)
        hr["HRV [ms]"] = hr["HRV [ms]"].map(hrv_comma_check)
        hr.meta["sensorID"] = sensorID

        out["hr"] = hr
    return out

def gpslogger_processing(directory, file, interpolation=None, interval=(0,3)):
    numeric = ["latitude","longitude","accuracy(m)","altitude(m)","geoid_height(m)","speed(m/s)","bearing(deg)"]
    dtformat = '%Y-%m-%d %H:%M:%S'
    df = pd.read_csv(directory+file,  index_col="date time",
                     date_format={'date time': dtformat})
    if interpolation !=None:
        df = interpolate(df, 1, interpolation, pd.Timedelta(
            '00:0:10'), numeric_columns=numeric, add_binary_counter=False)
        df = keep_interval(df, interval)
    return Apm(df)

