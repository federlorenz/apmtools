import os as os
import uuid as uuid
from .classes import DictionaryPlus
from .data_processing import upas_processing, lascar_processing, purple_processing
from typing import Dict, Tuple, List
import math
import bokeh.plotting as bopl
import numpy as np
import copy
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.models import (LinearAxis, Range1d)
from bokeh.palettes import Dark2_5 as palette
import itertools
from datetime import timedelta

def show(dictionary, number=0):
    """
    return an element of a dictionary
    If number is not specified, returns the values associated with the first key
    """
    try:
        return(dictionary[list(dictionary.keys())[number]])
    except:
        print("something's wrong")

def subset(dictionary, filter_dict, filter_style='all', condition=None):
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
    return_dict = copy.deepcopy(dictionary)

    if filter_style == 'any':
        a = {}
        for key, value in return_dict.items():
            for i, j in filter_dict.items():
                if hasattr(value, 'm') & (type(value.m) == type({})) & (i in value.m.keys()):
                    try:
                        if type(j) == type(""):
                            if eval("value.__getattr__('m')[\""+i+"\"]" + j):
                                a[key] = value
                                break
                        else:
                            if getattr(value,'m')[i] in j:
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
                            if getattr(value,i) in j:
                                a[key] = value
                                break
                    except:
                        pass
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
                            if getattr(value,i) not in j:
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
                            if getattr(value,'m')[i] not in j:
                                del a[key]
                                break
                    except:
                        pass
                else:
                    del a[key]
                    break

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
            a.filter_key = dictionary.filter_key
            return a
        else:
            b = {}
            for key, value in a.items():
                if condition(value):
                    b[key] = value
            b = DictionaryPlus(b)
            b.filter_key = dictionary.filter_key
            return b
    else:
        a = DictionaryPlus(a)
        a.filter_key = dictionary.filter_key
        return a



    return a

def set_attrib(dictionary, attribute):
    """
    returns the set of attribute values for dictionary
    """
    return_set = set()
    for i in dictionary.values():
        if hasattr(i, 'm') & (type(i.m) == type({})) & (attribute in i.m.keys()):
            try:
                return_set.add(getattr(i,'m')[attribute])
            except:
                pass
        else:
            try:
                return_set.add(getattr(i,attribute))
            except:
                pass
    
    return return_set

def __scan(directory="", levels=[], level=0, monitor=None, levels_dict=None, gmt_timezone_shift=0, output=DictionaryPlus(), interpolate=False):

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
                print(f"file {directory}{j} is not associated with any monitor type. If this is a monitoring file, please make sure to place the file below a directory identifying the monitor type")
            else:
                match monitor:
                    case "upas":
                        try:
                            processed = upas_processing(
                                directory, file=j, interpolate_data=interpolate)
                            for k, v in levels_dict.items():
                                if v not in ["None","_","-"]:
                                    processed.m[k] = v
                            processed.m["rejected"] = False
                            processed.m["filename"] = j
                            output[str(uuid.uuid4())] = processed
                        except:
                            print(
                                f"processing of file {directory}{j} failed. Is this a {monitor} file?")
                    case "lascar":
                        try:
                            processed = lascar_processing(
                                directory, file=j, interpolate_data=interpolate)
                            for k, v in levels_dict.items():
                                if v not in ["None", "_", "-"]:
                                    processed.m[k] = v
                            processed.m["rejected"] = False
                            processed.m["filename"] = j
                            output[str(uuid.uuid4())] = processed
                        except:
                            print(
                                f"processing of file {directory}{j} failed. Is this a {monitor} file?")

        elif os.path.isdir(f"{directory}{j}"):
            monitor = match_monitor(j, monitor)
            if monitor == "purple" and (False not in set([os.path.isfile(f"{directory}{j}/{z}") for z in os.listdir(f"{directory}{j}/")])) and (len(os.listdir(f"{directory}{j}/"))>0):
                    try:

                        processed = purple_processing(
                            f"{directory}{j}/", timezone_shift=timedelta(hours=gmt_timezone_shift), interpolate_data=interpolate)
                        for k, v in levels_dict.items():
                            if v not in ["None", "_", "-"]:
                                processed.m[k] = v
                        processed.m["rejected"] = False
                        processed.m["filename"] = j
                        output[str(uuid.uuid4())] = processed
                    except:
                        print(
                            f"processing of purple air directory {directory}{j} failed.")
            else:
                levels_dict[levels[level]] = j
                __scan(directory=f"{directory}{j}/", levels=levels,
                       level=level+1, monitor=monitor, levels_dict=levels_dict, gmt_timezone_shift=gmt_timezone_shift, output=output)

def scan(directory="", levels=[], level=0, monitor=None, levels_dict=None, gmt_timezone_shift=0, output=DictionaryPlus(), interpolate=False):

    data = DictionaryPlus()
    __scan(directory=directory, levels=levels, level=level, monitor=monitor, levels_dict=levels_dict,
           gmt_timezone_shift=gmt_timezone_shift, output=data, interpolate=interpolate)
    return data
