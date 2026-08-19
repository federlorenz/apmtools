import pandas as pd
import numpy as np
import copy

import itertools
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.layouts import column, layout
import bokeh.plotting as bopl
from bokeh.models.axes import DatetimeAxis, MercatorAxis
import os as os
import uuid as uuid
from datetime import timedelta

import xyzservices.providers as xyz


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
            attrib is an attribute of the elements of dictionary, and attrib_value is a list
            of the values of such attrib that the elements of returned dictionary can have
        specify filter_style='all' if all conditions should be met to be included in the return dictionary, specify filter_style='any' for including when any condition is met. Default is 'all'.
        """
        if type(filter_dict) != type(dict()):
            print("subset function error: type filter_dict should be dict")
            return
        return_dict = copy.deepcopy(self)
        a = {}

        if filter_style == 'all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) == type(""):
                                if not eval("value.__getattr__(\""+i+"\")" + j):
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
                            if type(j) == type(""):
                                if not eval("value.__getattr__('m')[\""+i+"\"]" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__('m')[\""+i+"\"]" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__(\""+i+"\")" + j):
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, i) in j:
                                    a[key] = value
                                    break
                        except:
                            pass

        if filter_style == 'negative':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) == type(""):
                                if eval("value.__getattr__(\""+i+"\")" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__('m')[\""+i+"\"]" + j):
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

    def len(self):
        a = len(self)
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

    @property
    def _constructor(self):
        return Dataset

    def apply_func(self, func, verbose=False):
        a = Dataset()
        for key, value in self.items():
            a[key] = func(value)
            if verbose:
                print(key)
        a.filter_key = self.filter_key
        return a

    def subset(self, filter_dict={}, filter_style='all', condition=None):
        """
        Return a subset of a DictionaryPlus, specified in the parameter filter_dict (itself a dictionary) or condition (a function that takes at minimum a value from the dictionary as an input parameter, and return True/False if some condition specified in the function is met. Typically a lambda function of the form lambda x: True if condition else False)
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

        if filter_style == 'all':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) == type(""):
                                if not eval("value.__getattr__(\""+i+"\")" + j):
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
                            if type(j) == type(""):
                                if not eval("value.__getattr__('m')[\""+i+"\"]" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__('m')[\""+i+"\"]" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__(\""+i+"\")" + j):
                                    a[key] = value
                                    break
                            else:
                                if getattr(value, i) in j:
                                    a[key] = value
                                    break
                        except:
                            pass

        if filter_style == 'negative':
            a = {key: value for key, value in return_dict.items()}
            for key, value in return_dict.items():
                for i, j in filter_dict.items():
                    if hasattr(value, i):
                        try:
                            if type(j) == type(""):
                                if eval("value.__getattr__(\""+i+"\")" + j):
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
                            if type(j) == type(""):
                                if eval("value.__getattr__('m')[\""+i+"\"]" + j):
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

        if condition != None:
            if a == {}:
                for key, value in return_dict.items():
                    if condition(value):
                        a[key] = value
                a = Dataset(a)
                a.filter_key = self.filter_key
                return a
            else:
                b = {}
                for key, value in a.items():
                    if condition(value):
                        b[key] = value
                b = Dataset(b)
                b.filter_key = self.filter_key
                return b
        else:
            a = Dataset(a)
            a.filter_key = self.filter_key
            return a

    def scan_folder(self, directory="./", levels=[], level=0, monitor=None, levels_dict=None, gmt_timezone_shift=0, interpolate=False):

        from .data_processing import upas_processing, lascar_processing, purple_processing

        if len(set(levels).intersection(set(["identifier", "start", "end", "length", "time", "pm25", "pm25correctedrh", "pm25correctedrhandgrav", "filterid", "grav_not", "grav"]))) > 0:
            print(
                f"the following levels names are not allowed: {set(levels).intersection(set(["identifier", "start", "end", "length", "time", "pm25", "pm25correctedrh", "pm25correctedrhandgrav", "filterid", "grav_not", "grav"]))}. Please choose different level names")
            return
        if levels_dict == None:
            levels_dict = dict(zip(levels, [None]*len(levels)))
        elements = os.listdir(directory)

        def match_monitor(x, monitor):
            if x.lower() in ["upas", "upass", "upas monitor"]:
                monitor = "upas"
            if x.lower() in ["lascar", "las"]:
                monitor = "lascar"
            if x.lower() in ["purple", "pair", "purple air", "purpleair", "purplea"]:
                monitor = "purple"
            else:
                monitor = monitor
            return monitor

        for j in elements:
            if (j.split(".")[-1] == "py") and (j != "py"):
                pass
            elif os.path.isfile(f"{directory}{j}"):
                if monitor == None:
                    print(f"file {directory}{j} is not associated with any monitor type. If this is a monitoring file, please make sure to place the file below a directory identifying the monitor type (i.e. a directory with the monitor name such as Upas, Lascar ...)")
                else:
                    match monitor:
                        case "upas":
                            try:
                                processed = upas_processing(
                                    directory, file=j, interpolate_data=interpolate)
                                for k, v in levels_dict.items():
                                    if v not in ["None", "_", "-", None]:
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                self[str(uuid.uuid4())] = processed
                            except:
                                print(
                                    f"processing of file {directory}{j} failed. Is this a {monitor} file?")
                        case "lascar":
                            try:
                                processed = lascar_processing(
                                    directory, file=j, interpolate_data=interpolate)
                                for k, v in levels_dict.items():
                                    if v not in ["None", "_", "-", None]:
                                        processed.m[k] = v
                                processed.m["rejected"] = False
                                processed.m["filename"] = j
                                self[str(uuid.uuid4())] = processed
                            except:
                                print(
                                    f"processing of file {directory}{j} failed. Is this a {monitor} file?")

            elif os.path.isdir(f"{directory}{j}"):
                monitor = match_monitor(j, monitor)
                if monitor == "purple" and (False not in set([os.path.isfile(f"{directory}{j}/{z}") for z in os.listdir(f"{directory}{j}/")])) and (len(os.listdir(f"{directory}{j}/")) > 0):
                    levels_dict[levels[level]] = j
                    try:
                        processed = purple_processing(
                            f"{directory}{j}/", timezone_shift=timedelta(hours=gmt_timezone_shift), interpolate_data=interpolate)
                        for k, v in levels_dict.items():
                            if v not in ["None", "_", "-", None]:
                                processed.m[k] = v
                        processed.m["rejected"] = False
                        processed.m["filename"] = j
                        self[str(uuid.uuid4())] = processed
                    except:
                        print(
                            f"processing of purple air directory {directory}{j} failed.")
                elif (len(os.listdir(f"{directory}{j}/")) == 0):
                    pass
                else:
                    levels_dict[levels[level]] = j
                    if level+1 < len(levels):
                        for r in range(level+1, len(levels_dict)):
                            levels_dict[levels[r]] = None
                    self.scan_folder(directory=f"{directory}{j}/", levels=levels,
                              level=level+1, monitor=monitor, levels_dict=levels_dict, gmt_timezone_shift=gmt_timezone_shift, interpolate=interpolate)

    def save_summary(self, save_csv=True):
        types_in = list(set(type(v) for v in self.values()))

        def match_class(x):
            if x is Upas:
                return ("Upas", run_upas())
            elif x is Lascar:
                return ("Lascar", run_lascar())
            elif x is Purple:
                return ("Purple", run_purple())
            # elif x is Apm:
            #     return "Apm"
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
                case Purple():
                    return "Purple"
                case Apm():
                    return "Apm"
                case PolarH10():
                    return "PolarH10"
                case Sum():
                    return "Sum"

        def run_upas():
            a = 0.524
            b = -0.0862
            c = 5.75
            d = self.subset(condition=lambda x: match_monitor(x) == "Upas")
            columns1 = ["identifier", "start", "end", "length", "time"]
            columns2 = list(set(d.meta()).difference(
                set(["filter", "header", "parameters"])))
            columns3 = ["pm25", "pm25correctedrh",
                        "pm25correctedrhandgrav", "filterid", "grav_not", "grav"]

            df = pd.DataFrame(columns=columns1+columns2+columns3)
            count = 0
            for k, v in d.items():
                print(f"processing UPAS {count+1} out of {len(d)}")
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
                    app.append(start)
                    app.append(end)
                    app.append(length)
                    app.append(v.index[i])
                    for m in columns2:
                        app.append(v.m[m] if v.m[m] != None else "")
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
                    df.loc[len(df)] = app
                count += 1
            return df

        def run_lascar():
            d = self.subset(condition=lambda x: match_monitor(x) == "Lascar")
            columns1 = ["identifier", "start", "end", "length", "time"]
            columns2 = list(set(d.meta()))
            columns3 = ["CO(ppm)"]

            df = pd.DataFrame(columns=columns1+columns2+columns3)
            count = 0
            for k, v in d.items():
                print(f"processing Lascar {count+1} out of {len(d)}")
                v = v.dropna(subset="CO(ppm)")
                identifier = k
                start = v.start
                end = v.end
                length = v.length
                for i in range(len(v)):
                    app = []
                    app.append(identifier)
                    app.append(start)
                    app.append(end)
                    app.append(length)
                    app.append(v.index[i])
                    for m in columns2:
                        app.append(v.m[m] if v.m[m] != None else "")
                    app.append(v["CO(ppm)"].iloc[i])
                    df.loc[len(df)] = app
                count += 1
            return df

        def run_purple():
            d = self.subset(condition=lambda x: match_monitor(x) == "Purple")
            columns1 = ["identifier", "start", "end", "length", "time"]
            columns2 = list(set(d.meta()))
            columns3 = ["pm2_5", "pm2_5_adj"]

            df = pd.DataFrame(columns=columns1+columns2+columns3)
            count = 0
            for k, v in d.items():
                print(f"processing Purple {count+1} out of {len(d)}")
                v = v.dropna(subset="pm_adj")
                identifier = k
                start = v.start
                end = v.end
                length = v.length
                for i in range(len(v)):
                    app = []
                    app.append(identifier)
                    app.append(start)
                    app.append(end)
                    app.append(length)
                    app.append(v.index[i])
                    for m in columns2:
                        app.append(v.m[m] if v.m[m] != None else "")
                    app.append(((v["pm2_5_cf_1"]+v["pm2_5_cf_1_b"])/2).iloc[i])
                    app.append(v["pm_adj"].iloc[i])
                    df.loc[len(df)] = app
                count += 1
            return df
        out = {}
        for cl in types_in:
            z = match_class(cl)
            if save_csv:
                z[1].to_csv(f"{z[0]}.csv", index=False)
            out[z[0]] = z[1]
        return out

    def save_data(self, directory="./saved/", levels=[]):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
        for k, v in self.items():
            for i in range(len(levels)):
                folder = [v.m[levels[j]]
                          for j in range(0, i+1) if levels[j] in v.m.keys()]
                folder = "/".join(folder)
                try:
                    os.mkdir(f"{directory}/{folder}")
                except FileExistsError:
                    print(folder)
            filename = "_".join([v.m[level]
                                for level in levels if level in v.m.keys()])+"_"+str(v.start.date())+"_"+k
            v.to_csv(
                f"{directory}{"/".join([v.m[levels[j]] for j in range(0, len(levels)) if levels[j] in v.m.keys()])}/{filename}.csv")

    def save_upas_filter_summary(self, directory="./"):

        from .data_processing import upas_processing, lascar_processing, purple_processing

        def match_monitor(x):
            match x:
                case Upas():
                    return "Upas"
                case Lascar():
                    return "Lascar"
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
                app.append("")
                app.append("")
                app.append("")
                app.append("")
                app.append("")
                app.append("")
                app.append(float(v.m["parameters"]["SampledVolumeOffset"].strip(
                ))/1000 if "SampledVolumeOffset" in v.m["parameters"].keys() else float(v.m["parameters"]["SampledVolume"].strip())/1000)
                df.loc[len(df)] = app
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

    def add_metadata(self,metadata={}):
        for k,v in metadata.items():
            for value in self.values():
                value.m[k] = v

    def remove_metadata(self, metadata= {}):
        for k, v in metadata.items():
            for value in self.values():
                if (k in value.m.keys()) and (value.m[k] == v):
                    value.m[k] = None

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

    def date_time_filter(
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

    def date_time_filter(
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
            return (self["cooking_counter"].value_counts().max())*((self.index[1]-self.index[0]))

    @property
    def min_event_length(self):
        if len(self) == 0:
            return np.nan
        else:
            return (self["cooking_counter"].value_counts().min())*((self.index[1]-self.index[0]))

    @property
    def mean_event_length(self):
        if len(self) == 0:
            return np.nan
        else:
            return (self["cooking_counter"].value_counts().mean())*((self.index[1]-self.index[0]))

    @property
    def cooking_time_per_day(self):
        if len(self) == 0:
            return np.nan
        elif len(self["cooking_counter"].value_counts()) == 0:
            return pd.Timedelta("00:00:00")
        else:
            return ((self["cooking_counter"].value_counts().sum())*((self.index[1]-self.index[0])) / self.length) * \
                pd.Timedelta("24:00:00")

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

class PurpleSeries(ApmSeries):
    def __init__(self, *args, **kwargs):
        ApmSeries.__init__(self, *args, **kwargs)
    _metadata = ['m']

    @property
    def _constructor(self):
        return PurpleSeries

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

    def date_time_filter(
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
                out[key] = value.date_time_filter(
                    time_start=time_start, time_end=time_end, date_start=date_start, date_end=date_end, day=day)

        return out

class Plot():
    def __init__(self):
        self.all_figures = []
        self.colors = itertools.cycle(palette)
        self.range_start = None
        self.range_end = None

    def add_figure(self, title=None, figure_sizes=(800, 1400), x_axis_type="datetime"):
        """
        x_axis_type = 'datetime' (default)
        x_axis_type = 'mercator'

        """
        if x_axis_type == "datetime":
            self.all_figures.append(bopl.figure(height=figure_sizes[0], width=figure_sizes[1], tools=["box_zoom", 'reset', 'wheel_zoom', "pan", "box_select"],
                                                x_axis_type=x_axis_type, x_axis_location="below",
                                                background_fill_color="#efefef"))
        if x_axis_type == "mercator":
            self.all_figures.append(bopl.figure(height=figure_sizes[0], width=figure_sizes[1], tools=["box_zoom", 'reset', 'wheel_zoom', "pan", "box_select"],
                                                x_axis_type=x_axis_type, y_axis_type=x_axis_type,
                                                background_fill_color="#efefef"))
            self.all_figures[-1].add_tile(xyz.OpenStreetMap.Mapnik)
        if title == None:
            self.all_figures[-1].title.text = f"Figure {len(self.add_figures)}"
            self.all_figures[-1].title.align = "center"
            self.all_figures[-1].title.text_font_size = "25px"
        else:
            self.all_figures[-1].title.text = title
            self.all_figures[-1].title.align = "center"
            self.all_figures[-1].title.text_font_size = "25px"

    def add_data_time(self, datain: DictionaryPlus, variable, plotn=None, filterdict=None, label="", color=None):
        datain = datain.subset(filterdict) if filterdict != None else datain
        if len(datain) == 0:
            pass
        else:
            if color == None:
                color = next(self.colors)
            if self.range_start == None:
                self.range_start = min(datain.set_attrib('start'))
            else:
                self.range_start = min(
                    min(datain.set_attrib('start')), self.range_start)
            if self.range_end == None:
                self.range_end = max(datain.set_attrib('end'))
            else:
                self.range_end = max(
                    max(datain.set_attrib('end')), self.range_end)

            for value in datain.values():
                dates = np.array(value.index, dtype=np.datetime64)
                source = ColumnDataSource(data=dict(date=dates, close=value[variable]))
                if plotn == None:
                    x = self.all_figures[-1].line('date', 'close', source=source, alpha=0.7,
                                                  muted_alpha=0.05, legend_label=label, color=color)
                else:
                    x = self.all_figures[plotn].line('date', 'close', source=source, alpha=0.7,
                                                     muted_alpha=0.05, legend_label=label, color=color)

    def add_data_vertical(self, datain: DictionaryPlus, variable, range_variable,plotn=None, filterdict=None, label="", color=None, line_width=0.3):
        datain = datain.subset(filterdict) if filterdict != None else datain
        if len(datain) == 0:
            pass
        else:
            if color == None:
                color = next(self.colors)
            if self.range_start == None:
                self.range_start = min(datain.set_attrib('start'))
            else:
                self.range_start = min(
                    min(datain.set_attrib('start')), self.range_start)
            if self.range_end == None:
                self.range_end = max(datain.set_attrib('end'))
            else:
                self.range_end = max(
                    max(datain.set_attrib('end')), self.range_end)

            maximus = max([value[range_variable].max() for value in datain.values()])
            minimum = 0 if (0 < min([value[range_variable].min() for value in datain.values(
            )])) else min([value[range_variable].min() for value in datain.values()])

            for value in datain.values():
                if plotn == None:                
                    for s in value[variable].value_counts().index:
                        left = value.loc[value[variable]==s].index[0]
                        right = value.loc[value[variable]== s].index[-1]
                        x = self.all_figures[-1].quad(left=left,right=right, top=maximus,bottom=minimum, alpha=0.02,
                                                  muted_alpha=0.2, legend_label=label, fill_color=color, line_alpha=0)
                else:
                    for s in value[variable].value_counts().index:
                        left = value.loc[value[variable]==s].index[0]
                        right = value.loc[value[variable]== s].index[-1]
                        x = self.all_figures[plotn].quad(left=left,right=right, top=maximus,bottom=minimum, alpha=0.02,
                                                  muted_alpha=0.2, legend_label=label, fill_color=color, line_alpha=0)

    def lnglat_to_meters(self, longitude, latitude):
        """
        Projects the given (longitude, latitude) values into Web Mercator
        coordinates (meters East of Greenwich and meters North of the Equator).

        Longitude and latitude can be provided as scalars, Pandas columns,
        or Numpy arrays, and will be returned in the same form.  Lists
        or tuples will be converted to Numpy arrays.

        Examples:
        easting, northing = lnglat_to_meters(-40.71,74)

        easting, northing = lnglat_to_meters(np.array([-74]),np.array([40.71]))

        df=pandas.DataFrame(dict(longitude=np.array([-74]),latitude=np.array([40.71])))
        df.loc[:, 'longitude'], df.loc[:, 'latitude'] = lnglat_to_meters(df.longitude,df.latitude)
        """
        if isinstance(longitude, (list, tuple)):
            longitude = np.array(longitude)
        if isinstance(latitude, (list, tuple)):
            latitude = np.array(latitude)

        origin_shift = np.pi * 6378137
        easting = longitude * origin_shift / 180.0
        northing = np.log(np.tan((90 + latitude) * np.pi / 360.0)
                          ) * origin_shift / np.pi
        return (easting, northing)

    def add_data_geo(self, datain: DictionaryPlus, lat, lon, plotn=None, filterdict=None, label="", color=None, linked_timeseries=True):
        if color == None:
            color = next(self.colors)
        if len(datain) == 0:
            pass
        else:
            for value in datain.values():
                dates = np.array(value.index, dtype=np.datetime64)
                longitude, latitude = self.lnglat_to_meters(
                    value[lon], value[lat])
                source = ColumnDataSource(
                    data=dict(date=dates, lat=latitude, lon=longitude, dummy=[np.nan for i in range(len(dates))]))
                if plotn == None:
                    x = self.all_figures[-1].scatter(x='lon', y='lat', source=source, alpha=0.7,
                                                     muted_alpha=0.05, legend_label=label, color=color, size=10)
                else:
                    x = self.all_figures[plotn].scatter(x='lon', y='lat', source=source, alpha=0.7,
                                                        muted_alpha=0.05, legend_label=label, color=color)
        # if linked_timeseries:
        #     self.all_figures[0].line('date', 'dummy', source=source, alpha=0,
        #                                      muted_alpha=0)

    def finalize(self, axis_labels=False, plot_layout=None):
        datarange = DataRange1d(start=self.range_start-(self.range_end-self.range_start)/20,
                                end=self.range_end+(self.range_end-self.range_start)/20)
        for j in range(len(self.all_figures)):
            self.all_figures[j].add_layout(
                self.all_figures[j].legend[0], 'right')
            self.all_figures[j].legend.click_policy = "mute"
        self.all_figures[0].x_range = datarange
        for key, value in enumerate(self.all_figures):
            if key > 0:
                if type(value.xaxis[0]) == type(DatetimeAxis()):
                    self.all_figures[key].x_range = self.all_figures[0].x_range
        if axis_labels:
            for key, value in enumerate(self.all_figures):
                if type(value.xaxis[0]) == type(DatetimeAxis()):
                    self.all_figures[key].yaxis.axis_label = axis_labels[key]
                    self.all_figures[key].yaxis.axis_label_orientation = 'vertical'
                    self.all_figures[key].yaxis.axis_label_text_font_size = '10px'

        if plot_layout == None:
            self.layout = column(self.all_figures)
        else:
            self.layout = layout(plot_layout)

    def show(self):
        bopl.show(self.layout)

    def save(self, filename=os.getcwd()+'/interactive_plots.html'):
        bopl.save(self.layout, filename=filename)
