import os as os
import uuid as uuid
from .classes import DictionaryPlus
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

def scan(directory, function, extension, target_dictionary):
    for j in os.listdir(directory):
        if j.split('.')[-1] == extension:
            processed = function(directory, j)
            target_dictionary[str(uuid.uuid4())] = processed
        elif (len(j.split('.'))) == 1:
            d = directory+j+'/'
            scan(d, function, extension, target_dictionary)
        else:
            pass

