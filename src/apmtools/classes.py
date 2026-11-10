import pandas as pd
import numpy as np
import copy
import os as os
import uuid as uuid
from datetime import timedelta
from io import StringIO
from csv import writer
from dateutil.parser import parse
import functools
import inspect
import json
import operator
from functools import reduce

from .units import upas_units, lascar_units, purple_units, lcm_units
from .completeness import analyse_completeness, complete_interval_mask
from .history_codec import encode_value, decode_value, load_history, save_history as _save_history_file
from .column_expressions import evaluate_column_expression


_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt,
        "<=": operator.le, "==": operator.eq, "!=": operator.ne}

def _evaluate_condition(df, spec):
    """spec examples:
       {"column": "age", "op": ">=", "value": 18}
       {"column": "group", "op": "isin", "value": ["a", "b"]}
       {"column": "age", "op": "isna"}
       {"and": [spec, spec, ...]} / {"or": [...]} / {"not": spec}
    """
    if "and" in spec:
        return reduce(operator.and_, (_evaluate_condition(df, s) for s in spec["and"]))
    if "or" in spec:
        return reduce(operator.or_, (_evaluate_condition(df, s) for s in spec["or"]))
    if "not" in spec:
        return ~_evaluate_condition(df, spec["not"])
    col, op = df[spec["column"]], spec["op"]
    if op == "isin":
        return col.isin(spec["value"])
    if op == "isna":
        return col.isna()
    if op == "notna":
        return col.notna()
    return _OPS[op](col, spec["value"])

def _as_mask(df, m):
    """Condition dict, boolean Series or array -> clean boolean Series."""
    if isinstance(m, dict):
        m = _evaluate_condition(df, m)
    return pd.Series(m, index=df.index).fillna(False).astype(bool)

def recorded(func):
    """Run the method, then append the call to the result's `history`.
    Raises TypeError up front if the arguments can't be stored in JSON."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k != "self"}
        if isinstance(params.get("mapping"), dict):          # keep key types
            params["mapping"] = [[k, v] for k, v in params["mapping"].items()]
        try:
            params = encode_value(params)
        except TypeError as e:
            raise TypeError(
                f"{func.__name__}: arguments must be history-serializable "
                f"(no masks, Series or callables). {e}") from None

        result = func(self, *args, **kwargs)
        result.history = list(getattr(self, "history", None) or []) + [
            {"method": func.__name__, "args": params}]
        return result
    return wrapper

###

class DictionaryPlus(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.filter_key = None

    @property
    def _constructor(self):
        return DictionaryPlus

    def show(self, number=0, key=None):
        """
        return an element of a dictionary
        If number is not specified, returns the values associated with the first key
        """
        try:
            if key != None:
                return (self.subset({self.filter_key: [key]}).show())
            else:
                if type(number) == type(""):
                    return (self.subset({self.filter_key: [number]}).show())
                else:
                    return (self[list(self.keys())[number]])
        except:
            print("something's wrong")

    def subset(self, filter_dict={}, filter_style='all', condition=None):
        """
        Return a subset of a DictionaryPlus, specified in the parameter filter_dict (itself a dictionary) or condition (a function that takes at minimum a value from the dictionary as an input parameter, and return True/False if some condition specified in the function is met. Typically a lambda function of the form lambda x: True if condition else False)
        filter_dict is {attrib:["attrib_value_x","attrib_value_y",..]}, where 
            attrib is an attribute of the elements of dictionary, and attrib_value is a value or list
            of the values of such attrib that the elements of returned dictionary can have
        specify filter_style='all' if all conditions should be met to be included in the return dictionary, other filters allowed are 'any', 'negative all', 'negative any'.
        """
        if type(filter_dict) != type(dict()):
            print("subset function error: type filter_dict should be dict")
            return
        return_dict = copy.deepcopy(self)
        a = {}

        if filter_style not in {'all', 'any', 'negative all', 'negative any'}:
            return return_dict

        if filter_style == 'all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) not in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, i) not in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] not in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] not in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    else:
                        del a[key]
                        break

        if filter_style == 'any':
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] in [j]:
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, 'm')[i] in j:
                                    a[key] = value
                                    break
                        except:
                            pass
                    else:
                        try:
                            if type(j) is not list:
                                if getattr(value, i) in [j]:
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, i) in j:
                                    a[key] = value
                                    break
                        except:
                            pass

        if filter_style == 'negative all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, i) in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    else:
                        break

        if filter_style == 'negative any':
            b = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) not in [j]:
                                    del b[key]
                                    break
                            else:
                                if getattr(value, i) not in j:
                                    del b[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] not in [j]:
                                    del b[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] not in j:
                                    del b[key]
                                    break
                        except:
                            pass
                    else:
                        del b[key]
                        break
            a = {key: value for key, value in return_dict.items()
                 if key not in b.keys()}

        if condition != None:
            if a == {}:
                for key, value in return_dict.items():
                    if condition(value):
                        a[key] = value
                a = DictionaryPlus(a)
                a.filter_key = self.filter_key
                return a
            else:
                b = {}
                for key, value in a.items():
                    if condition(value):
                        b[key] = value
                b = DictionaryPlus(b)
                b.filter_key = self.filter_key
                return b
        else:
            a = DictionaryPlus(a)
            a.filter_key = self.filter_key
            return a

    def set_attrib(self, attribute):
        """
        returns the set of attribute values for dictionary
        """
        return_set = set()
        for i in self.values():
            if hasattr(i, 'm') & (type(i.m) is dict) & (attribute in i.m.keys()):
                try:
                    return_set.add(getattr(i, 'm')[attribute])
                except TypeError:
                    try:
                        for j in getattr(i, 'm')[attribute]:
                            return_set.add(j)
                    except:
                        pass
            elif hasattr(i, attribute):
                try:
                    return_set.add(getattr(i, attribute))
                except TypeError:
                    try:
                        for j in getattr(i, attribute):
                            return_set.add(j)
                    except:
                        pass
                except AttributeError:
                    pass
            else:
                pass
        return return_set

    def meta(self, listall=False):
        meta = set().union(
            *[set(i.m.keys()) for i in self.values()])
        if listall:
            return {key: self.set_attrib(key) for key in meta}
        else:
            return meta

    def apply_func(self, func, verbose=False):
        a = DictionaryPlus()
        for key, value in self.items():
            a[key] = func(value)
            if verbose:
                print(key)
        a.filter_key = self.filter_key
        return a

    def concat_var(self, variable=None):
        if variable != None:
            a = self.apply_func(lambda x: x[variable])
        else:
            a = self
        return pd.concat(a)

class Dataset(DictionaryPlus):

    def __init__(self, *args, **kwargs):
        DictionaryPlus.__init__(self, *args, **kwargs)
        self.filter_key = None
        self.description = ""

    @property
    def _constructor(self):
        return Dataset

    def show(self, number=0, key=None):
        """
        return an element of a dictionary
        If number is not specified, returns the values associated with the first key
        """
        try:
            if key != None:
                return (self.subset({self.filter_key: [key]}).show())
            else:
                if type(number) == type(""):
                    return (self.subset({self.filter_key: [number]}).show())
                else:
                    return (self[list(self.keys())[number]])
        except:
            print("something's wrong")

    def apply_func(self, func, verbose=False):
        a = Dataset()
        for key, value in self.items():
            a[key] = func(value)
            if verbose:
                print(key)
        a.filter_key = self.filter_key
        a.description = self.description + f"\n Applied function {func}"
        return a

    def subset(self, filter_dict={}, filter_style='all', condition=None):
        """
        Return a subset of a Dataset, specified in the parameter filter_dict (itself a dictionary) or condition (a function that takes at minimum a value from the dictionary as an input parameter, and return True/False if some condition specified in the function is met. Typically a lambda function of the form lambda x: True if condition else False)
        filter_dict is {attrib:["attrib_value_x","attrib_value_y",..]}, where 
            attrib is an attribute of the elements of dictionary, and attrib_value is a list
            of the values of such attrib that the elements of returned dictionary can have
        specify filter_style='all' if all conditions should be met to be included in the return dictionary, specify filter_style='any' for including when any condition is met. Default is 'all'.
        """
        if type(filter_dict) != type(dict()):
            print("subset function error: type filter_dict should be dict")
            return
        return_dict = copy.deepcopy(self)
        a = {}

        if filter_style not in {'all', 'any', 'negative all', 'negative any'}:
            return return_dict

        if filter_style == 'all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) not in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, i) not in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] not in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] not in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    else:
                        del a[key]
                        break

        if filter_style == 'any':
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] in [j]:
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, 'm')[i] in j:
                                    a[key] = value
                                    break
                        except:
                            pass
                    else:
                        try:
                            if type(j) is not list:
                                if getattr(value, i) in [j]:
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, i) in j:
                                    a[key] = value
                                    break
                        except:
                            pass

        if filter_style == 'negative all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, i) in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] in [j]:
                                    del a[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    else:
                        break

        if filter_style == 'negative any':
            b = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) is not list:
                                if getattr(value, i) not in [j]:
                                    del b[key]
                                    break
                            else:
                                if getattr(value, i) not in j:
                                    del b[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                        try:
                            if type(j) is not list:
                                if getattr(value, 'm')[i] not in [j]:
                                    del b[key]
                                    break
                            else:
                                if getattr(value, 'm')[i] not in j:
                                    del b[key]
                                    break
                        except:
                            pass
                    else:
                        del b[key]
                        break
            a = {key:value for key,value in return_dict.items() if key not in b.keys()}

        if condition != None:
            if a == {}:
                for key, value in return_dict.items():
                    if condition(value):
                        a[key] = value
                a = Dataset(a)
                a.filter_key = self.filter_key
                a.description = self.description + \
                    f"\nSubset by {filter_dict} on style {filter_style} by condition {condition}"
                return a
            else:
                b = {}
                for key, value in a.items():
                    if condition(value):
                        b[key] = value
                b = Dataset(b)
                b.filter_key = self.filter_key
                b.description = self.description + \
                    f"\nSubset by {filter_dict} on style {filter_style} by condition {condition}"
                return b
        else:
            a = Dataset(a)
            a.filter_key = self.filter_key
            a.description = self.description + \
                f"\nSubset by {filter_dict} on style {filter_style} by condition {condition}"
            return a

    def scan_folder(self, directory="./", levels=[], level=0, monitor=None, levels_dict=None, utc_timezone=0, interpolate=False, add_m={}, message_callback=None, signatures ={}):

        from .data_processing import upas_processing, lascar_processing, purple_processing, lcm_processing,csv_processing

        def emit(message):
            # Preserve the original console output; the callback is opt-in.
            print(message)
            if message_callback is not None:
                try:
                    message_callback(str(message))
                except Exception:
                    pass  # A UI listener must not terminate a filesystem scan.

        if len(set(levels).intersection(set(["identifier", "start", "end", "length", "datetime", "timedelta","pm25", "pm25correctedrh", "pm25correctedrhandgrav", "filterid", "grav_not", "grav","PM2.5","PM10","C02","NO2"]))) > 0:
            emit(
                f"the following levels names are not allowed: {set(levels).intersection(set(["identifier", "start", "end", "length", "datetime", "timedelta", "pm25", "pm25correctedrh", "pm25correctedrhandgrav", "filterid", "grav_not", "grav","PM2.5","PM10","C02","NO2"]))}. Please choose different level names")
            return
        if levels_dict == None:
            levels_dict = dict(
                zip(list(set(levels)), [None]*len(list(set(levels)))))
        elements = os.listdir(directory)

        def monitor_signature(f):
            try:
                with open(f, encoding="utf-8") as file:
                    line = file.readline().split(",")
                for key,value in signatures.items():
                    if value["signature"] == line:
                        return (key,value)
                match line:
                    case ['PARAMETERS','VALUE', 'UNITS/NOTES\n']:
                        return "upas"
                    case ["UTCDateTime", "mac_address", "firmware_ver", "hardware", "current_temp_f", "current_humidity", "current_dewpoint_f", "pressure", "adc", "mem", "rssi", "uptime", "pm1_0_cf_1", "pm2_5_cf_1", "pm10_0_cf_1", "pm1_0_atm", "pm2_5_atm", "pm10_0_atm", "pm2.5_aqi_cf_1", "pm2.5_aqi_atm", "p_0_3_um", "p_0_5_um", "p_1_0_um", "p_2_5_um", "p_5_0_um", "p_10_0_um", "pm1_0_cf_1_b", "pm2_5_cf_1_b", "pm10_0_cf_1_b", "pm1_0_atm_b", "pm2_5_atm_b", "pm10_0_atm_b", "pm2.5_aqi_cf_1_b", "pm2.5_aqi_atm_b", "p_0_3_um_b", "p_0_5_um_b", "p_1_0_um_b", "p_2_5_um_b", "p_5_0_um_b", "p_10_0_um_b", "gas\n"]:
                        return "purple"
                    case ["timestamp","date","sensorName","deviceId","resolution","PM2.5","PM10","CO2","NO2","Temperature","Humidity\n"]:
                        return "lcm"
                    case _:
                        pass
                match line[1:3]:
                    case ['VALUE', 'UNITS/NOTES\n']:
                        return "upas"
                    case ["mac_address", "firmware_ver"]:
                        return "purple"
                    case ['Time', 'CO(ppm)']:
                        return "lascar"
                    case ["date","sensorName"]:
                        return "lcm"
                    case _:
                        return None
            except:
                return None

        for j in elements:
            if (j.split(".")[-1] == "py") and (j != "py"):
                pass
            elif os.path.isfile(f"{directory}{j}"):
                if monitor_signature(f"{directory}{j}") == None:
                    emit(
                        f"file {directory}{j} is not associated with any monitor type.")
                else:
                    monitor = monitor_signature(f"{directory}{j}")
                    if isinstance(monitor,tuple):
                        try:
                            processed = csv_processing(directory, file=j, time_column=monitor[1]["time_column"], date_format=monitor[
                                                       1]["date_format"], monitor_name=monitor[0], timezone_shift=timedelta(hours=utc_timezone))
                            for k, v in levels_dict.items():
                                if (v not in ["None", "_", "-", None]) and (k not in ["monitor", "", None, "-", "_"]):
                                    processed.m[k] = v
                            processed.m["rejected"] = False
                            processed.m["filename"] = j
                            for key, value in add_m.items():
                                processed.m[key] = value
                            self[str(uuid.uuid4())] = processed
                        except:
                            emit(f"processing of file {directory}{j} failed. Is this a {monitor[0]} file?")
                    match monitor:
                        case "upas":
                            try:
                                processed = upas_processing(
                                    directory, file=j, interpolate_data=interpolate)
                                for k, v in levels_dict.items():
                                    if (v not in ["None", "_", "-", None]) and (k not in ["monitor", "", None, "-", "_"]):
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                for key, value in add_m.items():
                                    processed.m[key] = value
                                self[str(uuid.uuid4())] = processed
                            except:
                                emit(
                                    f"processing of file {directory}{j} failed. Is this a {monitor} file?")
                        case "lascar":
                            try:
                                processed = lascar_processing(
                                    directory, file=j, interpolate_data=interpolate)
                                for k, v in levels_dict.items():
                                    if (v not in ["None", "_", "-", None]) and (k not in ["monitor", "", None, "-", "_"]):
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                for key, value in add_m.items():
                                    processed.m[key] = value
                                self[str(uuid.uuid4())] = processed
                            except:
                                emit(
                                    f"processing of file {directory}{j} failed. Is this a {monitor} file?")
                        case "lcm":
                            try:
                                processed = lcm_processing(
                                    directory, file=j, interpolate_data=interpolate, timezone_shift=timedelta(hours=utc_timezone))
                                for k, v in levels_dict.items():
                                    if (v not in ["None", "_", "-", None]) and (k not in ["monitor", "", None, "-", "_"]):
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                for key, value in add_m.items():
                                    processed.m[key] = value
                                self[str(uuid.uuid4())] = processed
                            except:
                                emit(
                                    f"processing of file {directory}{j} failed. Is this a {monitor} file?")
                        case "purple":
                            try:
                                processed = purple_processing(
                                    f"{directory}/", timezone_shift=timedelta(hours=utc_timezone), interpolate_data=interpolate)
                                for k, v in levels_dict.items():
                                    if (v not in ["None", "_", "-", None]) and (k not in ["monitor", "", None, "-", "_"]):
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                for key, value in add_m.items():
                                    processed.m[key] = value
                                self[str(uuid.uuid4())] = processed
                                break
                            except:
                                emit(
                                    f"processing of purple air directory {directory}{j} failed.")
                                break

            elif os.path.isdir(f"{directory}{j}"):
                if (len(os.listdir(f"{directory}{j}/")) == 0):
                    pass
                else:
                    if level < len(levels):
                        levels_dict[levels[level]] = j
                    if level+1 < len(levels):
                        for r in range(level+1, len(levels)):
                            if levels[r] not in levels[:level+1]:
                                levels_dict[levels[r]] = None
                    self.scan_folder(directory=f"{directory}{j}/", levels=levels,
                                     level=level+1, monitor=monitor, levels_dict=levels_dict, utc_timezone=utc_timezone, interpolate=interpolate, add_m=add_m, message_callback=message_callback)

    def save_summary(self, save_csv=True, filename="./", all_variables=False, merged=True, message_callback=None):
        def emit(message):
            print(message)
            if message_callback is not None:
                try:
                    message_callback(str(message))
                except Exception:
                    pass  # Reporting cannot corrupt Summary construction.

        types_in = list(set(type(v) for v in self.values()))

        def match_class(x):
            if x is Upas:
                return ("upas", run_upas())
            elif x is Lascar:
                return ("lascar", run_lascar())
            elif x is Purple:
                return ("purple", run_purple())
            elif x is Lcm:
                return ("lcm", run_lcm())
            elif x is Apm:
                return ("apm", run_apm())
            # elif x is PolarH10:
            #     return "PolarH10"
            # elif x is Sum:
            #     return "Sum"

        def match_monitor(x):
            match x:
                case Upas():
                    return "Upas"
                case Lascar():
                    return "Lascar"
                case Lcm():
                    return "Lcm"
                case Purple():
                    return "Purple"
                case Apm():
                    return "Apm"
                case PolarH10():
                    return "PolarH10"
                case Sum():
                    return "Sum"

        def run_apm():
            d = self.subset(condition=lambda x: match_monitor(x) == "Apm")
            columns1 = ["identifier", "monitor",
                        "start", "end", "length", "datetime", "timedelta"]
            columns2 = list(set(d.meta()))
            columns3 = ["apm_"+i for i in list(d.set_attrib("columns"))]

            buffer = StringIO()
            with buffer as file:
                df = writer(file, delimiter=",")
                df.writerow(columns1+columns2+columns3)
                count = 0
                for k, v in d.items():
                    emit(f"processing Apm {count+1} out of {len(d)}")
                    v = v.dropna(how="all")
                    identifier = k
                    start = v.start
                    end = v.end
                    length = v.length
                    monitor_name = v.monitor
                    for i in range(len(v)):
                        app = []
                        app.append(identifier)
                        app.append(monitor_name)
                        app.append(start)
                        app.append(end)
                        app.append(length)
                        app.append(v.index[i])
                        app.append(v.index[i]-v.start)
                        for m in columns2:
                            if m in v.m.keys():
                                app.append(v.m[m] if v.m[m] != None else "")
                            else:
                                app.append("")
                        for col in columns3:
                            if col[4:] in v.columns:
                                app.append(v[col[4:]].iloc[i])
                            else:
                                app.append(None)
                        df.writerow(list(map(str, app)))
                    count += 1
                file.seek(0)
                out = pd.read_csv(file)
                out["start"] = pd.to_datetime(out["start"])
                out["end"] = pd.to_datetime(out["end"])
                out["length"] = pd.to_timedelta(out["length"])
                out["datetime"] = pd.to_datetime(out["datetime"])
                out["timedelta"] = pd.to_timedelta(out["timedelta"])
                return (Summary(out, metadata=set(columns2)))

        def run_upas():
            a = 0.524
            b = -0.0862
            c = 5.75
            d = self.subset(condition=lambda x: match_monitor(x) == "Upas")
            columns1 = ["identifier", "monitor",
                        "start", "end", "length", "datetime","timedelta"]
            columns2 = list(set(d.meta()).difference(
                set(["filter", "header", "parameters"])))
            columns3 = ["upas_pm25"] if not all_variables else ["upas_"+i for i in list(d.set_attrib("columns"))]+["upas_pm25"]
            columns4 = ["upas_pm25_RHcorrected",
                        "upas_pm25_RHgravcorrected", "upas_filterid", "upas_gravimetric", "upas_gravimetric_blankcorrected"]
            buffer = StringIO()
            with buffer as file:
                df = writer(file, delimiter=",")
                df.writerow(columns1+columns2+columns3+columns4)
                count = 0
                for k, v in d.items():
                    emit(f"processing UPAS {count+1} out of {len(d)}")
                    v = v.dropna(subset="PM2_5MC")
                    identifier = k
                    start = v.start
                    end = v.end
                    length = v.length
                    filterid = v.m["filter"].filterid
                    grav_not = v.m["filter"].concentration * \
                        1000 if v.m["filter"].concentration != None else ""
                    grav = v.m["filter"].concentration_corrected * \
                        1000 if v.m["filter"].concentration_corrected != None else ""
                    vmean = (a*v["PM2_5MC"]+b*v["AtmoRH"]+c).mean()
                    for i in range(len(v)):
                        app = []
                        app.append(identifier)
                        app.append("upas")
                        app.append(start)
                        app.append(end)
                        app.append(length)
                        app.append(v.index[i])
                        app.append(v.index[i]-v.start)
                        for m in columns2:
                            if m in v.m.keys():
                                app.append(v.m[m] if v.m[m] != None else "")
                            else:
                                app.append("")

                        for col in columns3[:-1]:
                            if col[5:] in v.columns:
                                app.append(v[col[5:]].iloc[i])
                            else:
                                app.append(None)
                        app.append(v["PM2_5MC"].iloc[i])
                        xx = (a*v["PM2_5MC"].iloc[i])+(b*v["AtmoRH"].iloc[i])+c
                        app.append(xx)
                        if grav != "":
                            app.append(xx*grav/vmean)
                        else:
                            if grav_not != "":
                                app.append(xx*grav_not/vmean)
                            else:
                                app.append("")
                        app.append(filterid)
                        app.append(grav_not)
                        app.append(grav)
                        df.writerow((list(map(str, app))))
                    count += 1
                file.seek(0)
                out = pd.read_csv(file)
                out["start"] = pd.to_datetime(out["start"])
                out["end"] = pd.to_datetime(out["end"])
                out["length"] = pd.to_timedelta(out["length"])
                out["datetime"] = pd.to_datetime(out["datetime"])
                out["timedelta"] = pd.to_timedelta(out["timedelta"])
                return (Summary(out, metadata=set(columns2)))

        def run_lascar():
            d = self.subset(condition=lambda x: match_monitor(x) == "Lascar")
            columns1 = ["identifier", "monitor",
                        "start", "end", "length", "datetime", "timedelta"]
            columns2 = list(set(d.meta()))
            columns3 = ["lascar_CO(ppm)"]

            buffer = StringIO()
            with buffer as file:
                df = writer(file, delimiter=",")
                df.writerow(columns1+columns2+columns3)
                count = 0
                for k, v in d.items():
                    emit(f"processing Lascar {count+1} out of {len(d)}")
                    v = v.dropna(subset="CO(ppm)")
                    identifier = k
                    start = v.start
                    end = v.end
                    length = v.length
                    for i in range(len(v)):
                        app = []
                        app.append(identifier)
                        app.append("lascar")
                        app.append(start)
                        app.append(end)
                        app.append(length)
                        app.append(v.index[i])
                        app.append(v.index[i]-v.start)
                        for m in columns2:
                            if m in v.m.keys():
                                app.append(v.m[m] if v.m[m] != None else "")
                            else:
                                app.append("")
                        app.append(v["CO(ppm)"].iloc[i])
                        df.writerow(list(map(str, app)))
                    count += 1
                file.seek(0)
                out = pd.read_csv(file)
                out["start"] = pd.to_datetime(out["start"])
                out["end"] = pd.to_datetime(out["end"])
                out["length"] = pd.to_timedelta(out["length"])
                out["datetime"] = pd.to_datetime(out["datetime"])
                out["timedelta"] = pd.to_timedelta(out["timedelta"])
                return (Summary(out, metadata=set(columns2)))

        def run_lcm():
            a = 0.524
            b = -0.0862
            c = 5.75
            d = self.subset(condition=lambda x: match_monitor(x) == "Lcm")
            columns1 = ["identifier", "monitor",
                        "start", "end", "length", "datetime", "timedelta"]
            columns2 = list(set(d.meta()))
            columns3 = ["lcm_sensorName", "lcm_deviceId", "lcm_resolution",
                        "lcm_PM2.5", "lcm_PM10", "lcm_CO2", "lcm_NO2", "lcm_Temperature", "lcm_Humidity"] if all_variables else ["lcm_PM2.5"]
            columns4 = ["lcm_PM2.5correctedrh"]

            buffer = StringIO()
            with buffer as file:
                df = writer(file, delimiter=",")
                df.writerow(columns1+columns2+columns3+columns4)
                count = 0
                for k, v in d.items():
                    emit(f"processing Lcm {count+1} out of {len(d)}")
                    v = v.dropna(subset="PM2.5")
                    identifier = k
                    start = v.start
                    end = v.end
                    length = v.length
                    for i in range(len(v)):
                        app = []
                        app.append(identifier)
                        app.append("lcm")
                        app.append(start)
                        app.append(end)
                        app.append(length)
                        app.append(v.index[i])
                        app.append(v.index[i]-v.start)
                        for m in columns2:
                            if m in v.m.keys():
                                app.append(v.m[m] if v.m[m] != None else "")
                            else:
                                app.append("")
                        for m in columns3:
                            app.append(v[m[4:]].iloc[i])
                        xx = (a*v["PM2.5"].iloc[i])+(b*v["Humidity"].iloc[i])+c
                        app.append(xx)
                        df.writerow(list(map(str, app)))
                    count += 1
                file.seek(0)
                out = pd.read_csv(file)
                out["start"] = pd.to_datetime(out["start"])
                out["end"] = pd.to_datetime(out["end"])
                out["length"] = pd.to_timedelta(out["length"])
                out["datetime"] = pd.to_datetime(out["datetime"])
                out["timedelta"] = pd.to_timedelta(out["timedelta"])
                return (Summary(out, metadata=set(columns2)))

        def run_purple():
            d = self.subset(condition=lambda x: match_monitor(x) == "Purple")
            columns1 = ["identifier", "monitor",
                        "start", "end", "length", "datetime", "timedelta"]
            columns2 = list(set(d.meta()))
            columns3 = ["purple_pm25", "purple_pm25_RHcorrected"]
            columns4 = [] if not all_variables else ["purple_"+i for i in list(d.set_attrib("columns"))]
            buffer = StringIO()
            with buffer as file:
                df = writer(file, delimiter=",")
                df.writerow(columns1+columns2+columns3+columns4)
                count = 0
                for k, v in d.items():
                    emit(f"processing Purple {count+1} out of {len(d)}")
                    v = v.dropna(subset="pm_adj")
                    identifier = k
                    start = v.start
                    end = v.end
                    length = v.length
                    for i in range(len(v)):
                        app = []
                        app.append(identifier)
                        app.append("purple")
                        app.append(start)
                        app.append(end)
                        app.append(length)
                        app.append(v.index[i])
                        app.append(v.index[i]-v.start)
                        for m in columns2:
                            if m in v.m.keys():
                                app.append(v.m[m] if v.m[m] != None else "")
                            else:
                                app.append("")
                        app.append(
                            ((v["pm2_5_cf_1"]+v["pm2_5_cf_1_b"])/2).iloc[i])
                        app.append(v["pm_adj"].iloc[i])
                        for col in columns4:
                                if col[6:] in v.columns:
                                        app.append(v[col[6:]].iloc[i])
                                else:
                                        app.append(None)
                        df.writerow(list(map(str, app)))
                    count += 1
                file.seek(0)
                out = pd.read_csv(file)
                out["start"] = pd.to_datetime(out["start"])
                out["end"] = pd.to_datetime(out["end"])
                out["length"] = pd.to_timedelta(out["length"])
                out["datetime"] = pd.to_datetime(out["datetime"])
                out["timedelta"] = pd.to_timedelta(out["timedelta"])
                return (Summary(out, metadata=set(columns2)))

        out = {}
        for cl in types_in:
            z = match_class(cl)
            if save_csv:
                with open(f"{z[0]+"_"+filename if filename != "./" else z[0]+".csv"}", "w", encoding="utf-8") as output:
                    z[1].to_csv(output, index=False)
            out[z[0]] = z[1]
        if merged:
            out_merged = pd.concat([out[k] for k in out.keys()]).reset_index()
            out_merged.description = self.description
            return out_merged
        return out

    def save_data(self, directory="./saved/", levels=[]):
        def get_specific_level(v, j):
            if hasattr(v, j):
                return getattr(v, j)
            else:
                if j in v.m.keys():
                    return v.m[j]
                else:
                    return None
        try:
            os.mkdir(directory)
        except FileExistsError:
            if input(f"Directory {directory} exists already. Do you want to overwrite? Press enter to continue, or any other key plus enter to stop") != "":
                return
        for k, v in self.items():
            for i in range(len(levels)):
                folder = []
                for j in range(0, i+1):
                    if hasattr(v, levels[j]):
                        folder.append(getattr(v, levels[j]))
                    else:
                        if levels[j] in v.m.keys():
                            folder.append(v.m[levels[j]])
                folder = "/".join(folder)
                try:
                    os.mkdir(f"{directory}/{folder}")
                except FileExistsError:
                    pass
            filename = "_".join([get_specific_level(v, level) for level in levels if get_specific_level(
                v, level) != None])+"_"+str(v.start.date())+"_"+k
            v.to_csv(
                f"{directory}{"/".join([get_specific_level(v, levels[j]) for j in range(0, len(levels)) if get_specific_level(v, levels[j]) != None])}/{filename}.csv")

    def save_upas_filter_summary(self, directory="./"):

        from .data_processing import upas_processing, lascar_processing, purple_processing

        def match_monitor(x):
            match x:
                case Upas():
                    return "Upas"
                case Lascar():
                    return "Lascar"
                case Lcm():
                    return "Lcm"
                case Purple():
                    return "Purple"
                case Apm():
                    return "Apm"
                case PolarH10():
                    return "PolarH10"
                case Sum():
                    return "Sum"

        types_in = list(set(type(v) for v in self.values()))
        if Upas in types_in:
            d = self.subset(condition=lambda x: match_monitor(x) == "Upas")
            columns1 = ["identifier", "start", "end", "length"]
            columns2 = list(set(d.meta()).difference(
                set(["filter", "header", "parameters"])))
            columns3 = ["filterid", "pre_weight [mg]", "pre_weightsd [mg]", "post_weight [mg]",
                        "post_weightsd [mg]", "blank weight [mg]", "sampled volume [m3]"]

            df = pd.DataFrame(columns=columns1+columns2+columns3)

            count = 0
            for k, v in d.items():
                app = []
                v = v.dropna(subset="PM2_5MC")
                app.append(k)
                app.append(v.start)
                app.append(v.end)
                app.append(v.length)
                for m in columns2:
                    app.append(v.m[m] if v.m[m] != None else "")
                app.append(
                    f"{v.m["filter"].filterid if v.m["filter"].filterid != None else ""}")
                app.append(
                    f"{v.m["filter"].pre_weight if v.m["filter"].pre_weight != None else ""}")
                app.append(
                    f"{v.m["filter"].pre_weightsd if v.m["filter"].pre_weightsd != None else ""}")
                app.append(
                    f"{v.m["filter"].post_weight if v.m["filter"].post_weight != None else ""}")
                app.append(
                    f"{v.m["filter"].post_weightsd if v.m["filter"].post_weightsd != None else ""}")
                app.append(
                    f"{v.m["filter"].blanks if v.m["filter"].blanks != None else ""}")
                app.append(float(v.m["parameters"]["SampledVolumeOffset"].strip(
                ))/1000 if "SampledVolumeOffset" in v.m["parameters"].keys() else float(v.m["parameters"]["SampledVolume"].strip())/1000)
                df.loc[len(df)] = app
                
            df = df.sort_values(axis=0,by="identifier")
            df.to_csv(f"{directory}filter_summary.csv", index=False)
        else:
            print("No Upas files detected in Dataset")

    def load_upas_filter_summary(self, directory="./", filename="filter_summary.csv"):

        def nw(x):
            if pd.isna(x):
                return None
            else:
                return (x)

        try:
            with open(f"{directory}{filename}", "rb") as file:
                frame = pd.read_csv(file)
        except:
            print(
                "file loading failed. Check that the correct file and directory have been specified.")
        for i in range(len(frame)):
            df = frame.iloc[i]
            k = df["identifier"]
            try:
                self[k].m["filter"].filterid = nw(df["filterid"])
                self[k].m["filter"].pre_weight = nw(
                    float(df["pre_weight [mg]"]))
                self[k].m["filter"].pre_weightsd = nw(float(
                    df["pre_weightsd [mg]"]))
                self[k].m["filter"].post_weight = nw(float(
                    df["post_weight [mg]"]))
                self[k].m["filter"].post_weightsd = nw(float(
                    df["post_weightsd [mg]"]))
                self[k].m["filter"].blanks = nw(float(df["blank weight [mg]"]))
                self[k].m["filter"].sampled_volume = nw(float(
                    df["sampled volume [m3]"]))
            except:
                print("loading data failed. Check that the file is filled in correctly")

    def add_metadata(self, metadata={}):
        for k, v in metadata.items():
            for value in self.values():
                value.m[k] = v

    def remove_metadata(self, metadata={}):
        for k, v in metadata.items():
            for value in self.values():
                if (k in value.m.keys()) and (value.m[k] == v):
                    value.m[k] = None

    def get_monitor(self,value):
        return self.subset({"monitor":[value]})

class Group(pd.DataFrame):

    _metadata = ['m']

    @property
    def _constructor(self):
        return Group

    def __init__(self, *args, metadata={}, variable=None, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)

        self.m = metadata

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, Group):
            self.m = other.m.copy()

        return self

class Summary(pd.DataFrame):

    _metadata = ['metadata', 'views', 'variable', 'description', "history"]

    @property
    def _constructor(self):
        return Summary

    def __init__(self, *args, metadata=set(), variable=None, description="",**kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)

        self.metadata = metadata
        self.variable = variable
        self.description = description
        self.views = None
        self.history = None

    @property
    def plots(self):
        """Backward-compatible alias for saved plot and table views."""
        return self.views

    @plots.setter
    def plots(self, value):
        self.views = value

    def __setattr__(self, name, value):
        """Assign Summary metadata without pandas' column-creation fallback.

        Older pickles may hold a stale instance-level ``_metadata``.  Always
        consult the class declaration for metadata names; ordinary DataFrame
        attributes and column-assignment semantics continue through pandas.
        """
        if name in ("plots", "views") and value is not None and not isinstance(value, list):
            raise TypeError("Summary.views must be None or a list")
        if name in type(self)._metadata or name == "plots":
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def __setstate__(self, state):
        """Restore DataFrame state and default metadata absent in older pickles."""
        super().__setstate__(state)
        # Some older pickles contain an instance-level _metadata list, which
        # masks the corrected class declaration and breaks future copies.
        self.__dict__.pop("_metadata", None)
        # Migrate saved configurations from older pickles transparently.
        legacy = self.__dict__.pop("plots", None)
        if "views" not in self.__dict__:
            object.__setattr__(self, "views", legacy)
        for name, default in (("views", None), ("variable", None),
                              ("description", ""), ("metadata", set()), ("history", None)):
            if name not in self.__dict__:
                object.__setattr__(self, name, default)

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, Summary):
            self.variable = getattr(other, "variable", None)
            self.metadata = getattr(other, "metadata", set()).copy()
            self.description = getattr(other, "description", "")
            self.views = getattr(other, "views", getattr(other, "plots", None))
            self.history = copy.deepcopy(getattr(other, "history", None))

        return self

    def _subset_impl(self, filter_dict=None, filter_style='all'):
        """Apply categorical filters without recording an internal history step."""
        if filter_dict is None:
            filter_dict = {}
        if not isinstance(filter_dict, dict):
            raise TypeError("subset: filter_dict must be a dictionary")
        if filter_style not in {'all', 'any', 'negative all', 'negative any'}:
            raise ValueError(f"subset: invalid filter_style {filter_style!r}")

        conditions = []
        for column, values in filter_dict.items():
            if not isinstance(values, list):
                values = [values]
            if column in self.columns:
                conditions.append(self[column].isin(values).to_numpy())

        if not conditions:
            result = self.copy(deep=False)
        else:
            conditions = pd.DataFrame(conditions, dtype=bool)
            if filter_style == 'all':
                mask = conditions.all(axis=0).to_numpy()
            elif filter_style == 'any':
                mask = conditions.any(axis=0).to_numpy()
            elif filter_style == 'negative all':
                mask = (~conditions).all(axis=0).to_numpy()
            else:  # negative any
                mask = (~conditions).any(axis=0).to_numpy()
            result = self.loc[mask].copy()

        result.views = None
        result.description = self.description + f"\nSubset by {filter_dict} on style {filter_style}"
        return result

    @recorded
    def subset(self, filter_dict=None, filter_style='all'):
        """Return a new Summary matching categorical filters, recording one step."""
        return self._subset_impl(filter_dict, filter_style)

    def show(self, number=0, key=None):
        """
        return an element of a dictionary
        If number is not specified, returns the values associated with the first key
        """
        try:
            if key != None:
                return self.subset({"identifier": [key]})
            else:
                if type(number) == type(""):
                    return self.subset({"identifier": list(set(self["identifier"]))[0]})
                else:
                    return self.subset({"identifier": list(set(self["identifier"]))[number]})
        except:
            print("something's wrong")

    @recorded
    def dtfilter(
            self,
            time_start=None,
            time_end=None,
            date_start=None,
            date_end=None,
            day=None):
        """Filter datetime values, recording one reproducible history step.

        Date intervals are start-inclusive and end-exclusive. With both time
        endpoints supplied, an overnight interval crosses midnight. Weekdays
        are ISO values 1 (Monday) through 7 (Sunday).
        """
        if isinstance(time_start, str):
            time_start = parse(time_start).time()
        if isinstance(time_end, str):
            time_end = parse(time_end).time()
        if isinstance(date_start, str):
            date_start = parse(date_start)
        if isinstance(date_end, str):
            date_end = parse(date_end)

        # Always start with a fresh mask so even no-op calls return a new object.
        mask = pd.Series(True, index=self.index)
        dtcol = self["datetime"]
        if date_start is not None:
            mask &= dtcol >= date_start
        if date_end is not None:
            mask &= dtcol < date_end
        if time_start is not None or time_end is not None:
            times = pd.to_datetime(dtcol).dt.time
            if time_start is not None and time_end is not None:
                if time_start > time_end:
                    mask &= ((times >= time_start) | (times < time_end))
                else:
                    mask &= ((times >= time_start) & (times < time_end))
            elif time_start is not None:
                mask &= times >= time_start
            else:
                # Preserve the existing end-only inclusive convention.
                mask &= times < time_end
        if day is not None:
            weekdays = pd.to_datetime(dtcol).dt.isocalendar().day
            mask &= weekdays.isin(day if isinstance(day, (list, tuple, set)) else [day])

        result = self.loc[mask.fillna(False).to_numpy()].copy().reset_index()
        result.description = (self.description +
            f"\nFiltered by {time_start if time_start is not None else ''} till "
            f"{time_end if time_end is not None else ''}, "
            f"{date_start if date_start is not None else ''} till "
            f"{date_end if date_end is not None else ''}, "
            f"{day if day is not None else ''}")
        result.views = None
        return result

    def set_attrib(self, attribute):
        """
        returns the set of attribute values for Summary
        """
        if attribute in self.columns:
            return set(self[attribute])

    def completeness_report(self, value_cols, frequency, *, time_col='datetime',
                            id_col='identifier', min_completeness=1.0,
                            include_empty_intervals=False):
        """Report observed/expected samples by interval and measurement.

        The result is an ordinary DataFrame with one row per identifier,
        interval and variable. See :func:`apmtools.completeness.analyse_completeness`.
        """
        return analyse_completeness(
            self, value_cols, frequency, time_col=time_col, id_col=id_col,
            min_completeness=min_completeness,
            include_empty_intervals=include_empty_intervals)

    def group_stat(self, value_cols, frequency=None, grouping_by=["identifier"], time_col='datetime', id_col='identifier',
                   min_completeness=1.0, only_complete=True, flooring_timestamp=True, aggregate=["mean"]):
        """
        Filter a DataFrame down to only the (monitor, hour) groups that have
        a full hour of data, then it's ready for an hourly groupby/mean.

        - Infers each monitor's sampling interval from its own median gap
        between readings (handles monitors with different cadences, e.g.
        5-min vs 2-min logging).
        - An hour is "complete" if its reading count is >= min_completeness
        * expected_count_for_that_monitor. Use 1.0 for strict full hours,
        or e.g. 0.9 to tolerate the odd dropped reading.

        Returns the filtered row-level DataFrame (not yet aggregated), so you
        can still group/aggregate however you like afterwards.
        """

        if len(self) < 1:
            print("Summary of no length")
            return None
        for v in value_cols:
            if v not in self.columns:
                print(f"{v} not in  Summary columns")
                return None
        if grouping_by != None:
            for v in grouping_by:
                if v not in self.columns:
                    print(f"{v} not in  Summary columns")
                    return None

        # Non-temporal grouping also works on previously aggregated frames.
        # Only operations involving a frequency need the source time column.
        if frequency is not None and time_col not in self.columns:
            raise KeyError(f"Temporal aggregation requires column {time_col!r}")
        df = (self.dropna(subset=[time_col]).copy()
              if frequency is not None else self.copy())

        if frequency != None:

            if only_complete:
                # Stage 4A: one shared completeness implementation for analysis
                # and aggregation. Retain existing all-value filtering semantics.
                df = df.loc[complete_interval_mask(
                    df, value_cols, frequency, time_col=time_col,
                    id_col=id_col, min_completeness=min_completeness)].copy()

            if isinstance(frequency, str):
                if len(frequency.split(" ")) == 1:
                    match frequency:
                        case "hour":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor("h")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.hour))
                        case "day":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor("d")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.day))
                        case "minute":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor("min")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.minute))
                        case "second":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor("s")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.second))
                        case "week":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor("w")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.week))
                        case "weekday":
                            df[frequency] = df["datetime"].map(
                                lambda x: (x.weekday()))

                if len(frequency.split(" ")) == 2:
                    mult = float(frequency.split(" ")[0])
                    match frequency.split(" ")[1]:
                        case "hour":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor(f"{mult}h")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.hour//mult))*mult
                        case "day":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor(
                                    f"{mult}d")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.day//mult))*mult
                        case "minute":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor(
                                    f"{mult}min")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.minute//mult))*mult
                        case "second":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor(
                                    f"{mult}s")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.second//mult))*mult
                        case "week":
                            if flooring_timestamp:
                                df[frequency] = df["datetime"].dt.floor(
                                    f"{mult}w")
                            else:
                                df[frequency] = df["datetime"].map(
                                    lambda x: (x.week//mult))*mult
                        case "weekday":
                            df[frequency] = df["datetime"].map(
                                lambda x: (x.weekday//mult))*mult

            if isinstance(frequency, list):
                for freq in frequency:
                    if len(freq.split(" ")) == 1:
                        match freq:
                            case "hour":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor("h")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.hour))
                            case "day":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor("d")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.day))
                            case "minute":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor("min")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.minute))
                            case "second":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor("s")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.second))
                            case "week":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor("w")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.week))
                            case "weekday":
                                df[freq] = df["datetime"].map(
                                    lambda x: (x.weekday()))

                    if len(freq.split(" ")) == 2:
                        mult = float(freq.split(" ")[0])
                        match freq.split(" ")[1]:
                            case "hour":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor(f"{mult}h")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.hour//mult))*mult
                            case "day":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor(
                                        f"{mult}d")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.day//mult))*mult
                            case "minute":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor(
                                        f"{mult}min")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.minute//mult))*mult
                            case "second":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor(
                                        f"{mult}s")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.second//mult))*mult
                            case "week":
                                if flooring_timestamp:
                                    df[freq] = df["datetime"].dt.floor(
                                        f"{mult}w")
                                else:
                                    df[freq] = df["datetime"].map(
                                        lambda x: (x.week//mult))*mult
                            case "weekday":
                                df[freq] = df["datetime"].map(
                                    lambda x: (x.weekday//mult))*mult

        # A frequency may also be an explicitly selected grouping variable.
        # pandas allows repeated group keys but reset_index() then attempts to
        # insert the same column twice (e.g. "cannot insert hour, already exists").
        # Preserve the requested order while grouping each key only once.
        grouping_keys = list(dict.fromkeys(
            list(grouping_by or []) +
            ([frequency] if isinstance(frequency, str) else
             list(frequency) if isinstance(frequency, list) else [])
        ))
        if frequency is not None or grouping_by is not None:
            output = df.groupby(grouping_keys, dropna=False) if aggregate is not None else df[grouping_keys]
        else:
            output = df

        if aggregate is not None:
            output = output[value_cols].agg(aggregate)

        if aggregate != None and (frequency != None or grouping_by != None):
            output.columns = [
                "_".join(str(x) for x in col if pd.notna(x) and x != "")
                for col in output.columns.to_flat_index()
            ]

        output = output.reset_index()

        output = Group(output)

        output.m["value_cols"] = value_cols
        output.m["frequency"] = frequency
        output.m["grouping_by"] = grouping_by
        output.m["aggregate"] = aggregate

        return output
    
    @recorded
    def get_monitor(self, value):
        """Select a monitor without separately recording the internal subset."""
        result = self._subset_impl({"monitor": [value]}, 'all')
        result = result.dropna(axis=1, how="all")
        result.views = None
        return result

    @recorded
    def change_values(self, column, mapping):
        if not isinstance(mapping, dict):
            mapping = {k: v for k, v in mapping}
        result = self.copy()
        datatype = result[column].dtype


        def map_value(x):
            try:
                return mapping.get(x, x)
            except TypeError:  # x is unhashable, so it can't be a key in mapping
                return x

        result[column] = result[column].map(map_value)

        try:
            result[column] = result[column].astype(datatype)
        except (ValueError, TypeError):
            pass

        result.views = None
        return result

    @recorded
    def map_values(self, target_column, mask, when_true, when_false=None, subset=None):
        """mask:   condition dict deciding True/False, e.g.
                   {"column": "age", "op": ">=", "value": 18}
        when_false: if None, rows where the mask is False keep their values.
        subset: optional condition dict; only rows satisfying it are eligible
                for change. Rows outside it always keep their original values.
        Mask and subset are both evaluated on the data as it was on entry.
        """
        result = self.copy()
        original = result[target_column]
        datatype = original.dtype

        mask = _as_mask(result, mask)
        scope = (_as_mask(result, subset) if subset is not None
                 else pd.Series(True, index=result.index))

        if when_false is None:
            new_values = original.where(~(mask & scope), when_true)
        else:
            new_values = pd.Series(
                np.where(mask, when_true, when_false), index=result.index
            ).where(scope, original)

        result[target_column] = new_values
        try:
            result[target_column] = result[target_column].astype(datatype)
        except (ValueError, TypeError):
            pass

        result.views = None
        return result

    @recorded
    def edit_columns(self, action, column, new_name=None, fill=None):
        """Return a modified copy of the DataFrame; the original is never changed.

        action:   "add" | "rename" | "remove" | "duplicate"
        column:   add       -> name of the new column
                rename    -> existing column to rename
                remove    -> existing column to drop
                duplicate -> existing column to copy
        new_name: required for "rename" and "duplicate"

        Invalid name situations fail quietly: the unchanged copy is returned.
        """
        result = self.copy()
        cols = result.columns

        if action == "add":
            if column in cols:
                return result
            result[column] = fill                      # appended at the end

        elif action == "rename":
            if column not in cols or new_name is None or new_name in cols:
                return result
            result = result.rename(columns={column: new_name})

        elif action == "remove":
            if column not in cols:
                return result
            result = result.drop(columns=[column])

        elif action == "duplicate":
            if column not in cols or new_name is None or new_name in cols:
                return result
            result[new_name] = result[column].copy()     # appended at the end

        else:
            return result

        result.views = None
        return result

    @recorded
    def calculate_column(self, name, expression):
        """Create or replace a column using a restricted vector expression.

        Examples::

            summary.calculate_column('weighted', 'a * 0.7 + b * 0.3')
            summary.calculate_column('flag', 'where(a > 10, 1, 0)')
            summary.calculate_column('ratio', "col('PM 2.5') / b")

        This returns an independent Summary and records JSON-replayable history.
        Only explicitly approved mathematical functions and operators are allowed.
        """
        if not isinstance(name, str) or not name:
            raise ValueError('name must be a non-empty column name')
        if not self.columns.is_unique:
            raise ValueError('calculate_column requires unique column names')
        # Evaluate against original data before creating/replacing the target.
        values = evaluate_column_expression(self, expression)
        result = self.copy()
        result[name] = values
        result.views = None
        return result

    # ---- serialization -------------------------------------------------
    def save_history(self, path):
        """Save transformation history as strict, versioned JSON."""
        _save_history_file(path, self.history or [])

    def apply_history(self, source):
        """Replay approved recorded transformations from a path or step list.

        History describes explicitly recorded methods. Arbitrary pandas
        slicing and other non-recorded operations are not captured.
        """
        steps = load_history(source)   # Validate *all* steps before execution.
        validated = []
        for index, step in enumerate(steps):
            method = getattr(self, step["method"])
            kwargs = decode_value(step["args"])
            try:
                inspect.signature(method).bind(**kwargs)
            except TypeError as exc:
                raise ValueError(f"Invalid arguments for history step {index}: {exc}") from exc
            validated.append((step["method"], kwargs))

        result = self
        for method_name, kwargs in validated:
            result = getattr(result, method_name)(**kwargs)
        return result

class Apm(pd.DataFrame):
    monitor = "apm"
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.m = {}
        self.variable = None
    _metadata = ['m']

    @property
    def _constructor(self):
        return Apm

    @property
    def _constructor_sliced(self):
        return ApmSeries

    @property
    def units(self):
        return {}

    @property
    def end(self):
        if len(self) == 0:
            return np.nan
        else:
            return self.index[-1]

    @property
    def start(self):
        if len(self) == 0:
            return np.nan
        else:
            return self.index[0]

    @property
    def length(self):
        if len(self) == 0:
            return np.nan
        else:
            return len(self)*(self.index[1]-self.index[0])

    def dtfilter(
            self,
            time_start=None,
            time_end=None,
            date_start=None,
            date_end=None,
            day=None):
        """Filters a file by time or date\n
            Input time as dt.time(hrs,min), and date as dt.date(year,month,day),\n
            and day as [1,2] list of days, with 1 Monday and 7 Sunday,\n
            if selecting a specific date interval that includes time, just specify\n
            that as dt.datetime interval under date_start and date_end"""
        if type(time_start) is str:
            time_start = parse(time_start).time()
        if type(time_end) is str:
            time_end = parse(time_end).time()
        if type(date_start) is str:
            date_start = parse(date_start)
        if type(date_end) is str:
            date_end = parse(date_end)

        if date_start is not None:
            self = self.loc[self.index >= date_start]
        if date_end is not None:
            self = self.loc[self.index < date_end]
        if (time_start is not None) & (time_end is not None):
            if time_start > time_end:
                self = self.loc[(self.index.time >= time_start)
                                | (self.index.time < time_end)]
            else:
                self = self.loc[(self.index.time >= time_start)
                                & (self.index.time < time_end)]
        if (time_start is not None) & (time_end is None):
            self = self.loc[self.index.time >= time_start]
        if (time_start is None) & (time_end is not None):
            self = self.loc[self.index.time <= time_end]
        if day is not None:
            self = self.loc[[a in day for a in [self.index[i].date().isoweekday()
                                                for i in range(len(self.index))]]]

        return self

    def func(self,function=lambda x:x, variable=None):
        if self.variable == None:
            if variable==None:
                return None
            else:
                return function(self[variable].dropna())
        else:
            return function(self[self.variable].dropna())       

class ApmSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        pd.Series.__init__(self, *args, **kwargs)
        
    _metadata = ['m']
    
    @property
    def _constructor(self):
        return ApmSeries

    @property
    def end(self):
        if len(self) == 0:
            return np.nan
        else:
            return self.index[-1]

    @property
    def start(self):
        if len(self) == 0:
            return np.nan
        else:
            return self.index[0]

    @property
    def length(self):
        if len(self) == 0:
            return np.nan
        else:
            return len(self)*(self.index[1]-self.index[0])

    def dtfilter(
            self,
            time_start=None,
            time_end=None,
            date_start=None,
            date_end=None,
            day=None):
        """Filters a file by time or date\n
            Input time as dt.time(hrs,min), and date as dt.date(year,month,day),\n
            and day as [1,2] list of days, with 1 Monday and 7 Sunday,\n
            if selecting a specific date interval that includes time, just specify\n
            that as dt.datetime interval under date_start and date_end"""
        if type(time_start) is str:
            time_start = parse(time_start).time()
        if type(time_end) is str:
            time_start = parse(time_end).time()
        if type(date_start) is str:
            date_start = parse(date_start)
        if type(date_end) is str:
            date_end = parse(date_end)
            
        if date_start is not None:
            self = self.loc[self.index >= date_start]
        if date_end is not None:
            self = self.loc[self.index < date_end]
        if (time_start is not None) & (time_end is not None):
            if time_start > time_end:
                self = self.loc[(self.index.time >= time_start)
                                | (self.index.time < time_end)]
            else:
                self = self.loc[(self.index.time >= time_start)
                                & (self.index.time < time_end)]
        if (time_start is not None) & (time_end is None):
            self = self.loc[self.index.time >= time_start]
        if (time_start is None) & (time_end is not None):
            self = self.loc[self.index.time <= time_end]

        if day is not None:
            self = self.loc[[a in day for a in [self.index[i].date().isoweekday()
                                                for i in range(len(self.index))]]]

        return self

class Sum(Apm):
    monitor = "sum"
    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.m = {}
        self.sampling_rate = pd.Timedelta(seconds=300)
    _metadata = ['m']

    @property
    def _constructor(self):
        return Sum

    @property
    def _constructor_sliced(self):
        return SumSeries

    @property
    def number_of_events(self):
        if len(self) == 0:
            return np.nan
        else:
            return len(self["cooking_counter"].value_counts())

    @property
    def max_event_length(self):
        if len(self) == 0:
            return np.nan
        else:
            return (self["cooking_counter"].value_counts().max())*((self.sampling_rate))

    @property
    def min_event_length(self):
        if len(self) == 0:
            return np.nan
        else:
            return (self["cooking_counter"].value_counts().min())*((self.sampling_rate))

    @property
    def mean_event_length(self):
        if len(self) == 0:
            return np.nan
        else:
            return (self["cooking_counter"].value_counts().mean())*((self.sampling_rate))

    @property
    def cooking_time(self):
        if len(self) == 0:
            return np.nan
        elif len(self["cooking_counter"].value_counts()) == 0:
            return pd.Timedelta("00:00:00")
        else:
            return (self["cooking_counter"].value_counts().sum())*((self.sampling_rate))

    @property
    def cooking_time_per_day(self):
        if len(self) == 0:
            return np.nan
        elif len(self["cooking_counter"].value_counts()) == 0:
            return pd.Timedelta("00:00:00")
        else:
            return ((self["cooking_counter"].value_counts().sum())*((self.sampling_rate)) / self.length) * \
                pd.Timedelta("24:00:00")

    @property
    def cooking_events(self):
        if len(self) == 0:
            return np.nan
        elif len(self["cooking_counter"].value_counts()) == 0:
            return 0
        else:
            return self.number_of_events

    @property
    def cooking_events_per_day(self):
        if len(self) == 0:
            return np.nan
        elif len(self["cooking_counter"].value_counts()) == 0:
            return 0
        else:
            return self.number_of_events / \
                (self.length.total_seconds() / (3600 * 24))

class SumSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return SumSeries

class Grav_Filter():

    def __init__(self, *args, **kwargs):
        self.filterid = None
        self.pre_weight = None
        self.pre_weightsd = None
        self.post_weight = None
        self.post_weightsd = None
        self.blanks = None
        self.sampled_volume = None
        self.concentration_manual_input = None

    @property
    def difference(self):
        if (self.pre_weight == None) | (self.post_weight == None):
            return None
        else:
            return self.post_weight - self.pre_weight

    @property
    def difference_corrected(self):
        if (self.difference == None) | (self.blanks == None):
            return None
        else:
            return self.difference - self.blanks

    @property
    def concentration(self):
        if (self.difference == None) | (self.sampled_volume == None):
            return None
        else:
            return self.difference/self.sampled_volume

    @property
    def concentration_corrected(self):
        if (self.difference_corrected == None) | (self.sampled_volume == None):
            return None
        else:
            return self.difference_corrected/self.sampled_volume

class Upas(Apm):
    monitor = "upas"
    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.m = {}
        self.m["filter"] = Grav_Filter()
        self.variable = "PM2_5MC"
    _metadata = ['m']

    @property
    def _constructor(self):
        return Upas

    @property
    def _constructor_sliced(self):
        return UpasSeries

    @property
    def units(self):
        return upas_units

class UpasSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return UpasSeries

class Lascar(Apm):
    monitor = "lascar"
    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.m = {}
        self.variable = "CO(ppm)"
    _metadata = ['m']
    
    @property
    def _constructor(self):
        return Lascar

    @property
    def _constructor_sliced(self):
        return LascarSeries

    @property
    def units(self):
        return lascar_units

class LascarSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return LascarSeries

class Purple(Apm):
    monitor = "purple"
    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.m = {}
        self.variable = "pm_adj"
    _metadata = ['m']

    @property
    def _constructor(self):
        return Purple

    @property
    def _constructor_sliced(self):
        return PurpleSeries

    @property
    def units(self):
        return purple_units

class PurpleSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return PurpleSeries

class Lcm(Apm):
    monitor = "lcm"

    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.m = {}
        self.variable = "PM2.5"

    @property
    def _constructor(self):
        return Lcm

    @property
    def _constructor_sliced(self):
        return LcmSeries

    @property
    def units(self):
        return lcm_units

class LcmSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return LcmSeries

class PolarH10(dict):
    monitor = "polarh10"
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self["ecg"] = None
        self["acc"] = None
        self["rr"] = None
        self["hr"] = None
        self.m = {}
    _metadata = ['m']

    @property
    def end(self):
        if not any([True if type(i) != type(None) else False for i in self.values()]):
            return np.nan
        else:
            return {key: value.end for key, value in self.items() if type(value) != type(None)}

    @property
    def start(self):
        if not any([True if type(i) != type(None) else False for i in self.values()]):
            return np.nan
        else:
            return {key: value.start for key, value in self.items() if type(value) != type(None)}

    @property
    def length(self):
        if not any([True if type(i) != type(None) else False for i in self.values()]):
            return np.nan
        else:
            return {key: value.length for key, value in self.items() if type(value) != type(None)}

    def dtfilter(
            self,
            time_start=None,
            time_end=None,
            date_start=None,
            date_end=None,
            day=None):
        """Filters a file by time or date\n
            Input time as dt.time(hrs,min), and date as dt.date(year,month,day),\n
            and day as [1,2] list of days, with 1 Monday and 7 Sunday,\n
            if selecting a specific date interval that includes time, just specify\n
            that as dt.datetime interval under date_start and date_end"""

        out = copy.deepcopy(self)

        for key, value in out.items():
            if type(value) != type(None):
                out[key] = value.dtfilter(
                    time_start=time_start, time_end=time_end, date_start=date_start, date_end=date_end, day=day)

        return out
