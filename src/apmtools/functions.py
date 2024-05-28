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

def subset(dictionary, filter_dict, filter_style='all'):
    """
    Return a subset of a dictionary, specified in filter_dict (itself a dictionary)
    filter_dict is {attrib:["attrib_value_x","attrib_value_y",..]} or {attrib:"condition"}, where 
        attrib is an attribute of the elements of dictionary, and attrib_value is a list
        of the values of such attrib that the elements of returned dictionary can have, and condition    
        is the string of the condition that the attribute should verify, such as for example "< 0"
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
                if hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                    try:
                        if type(j) == type(""):
                            if eval("value.__getattr__('meta')[\""+i+"\"]" + j):
                                a[key] = value
                                break
                        else:
                            if value.__getattr__('meta')[i] in j:
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
                            if value.__getattr__(i) in j:
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
                            if value.__getattr__(i) not in j:
                                del a[key]
                                break
                    except:
                        pass
                elif hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                    try:
                        if type(j) == type(""):
                            if not eval("value.__getattr__('meta')[\""+i+"\"]" + j):
                                del a[key]
                                break
                        else:
                            if value.__getattr__('meta')[i] not in j:
                                del a[key]
                                break
                    except:
                        pass
                else:
                    del a[key]
                    break

    return a

def set_attrib(dictionary, attribute):
    """
    returns the set of attribute values for dictionary
    """
    return_set = set()
    for i in dictionary.values():
        if hasattr(i, 'meta') & (type(i.meta) == type({})) & (attribute in i.meta.keys()):
            try:
                return_set.add(i.__getattr__('meta')[attribute])
            except:
                pass
        else:
            try:
                return_set.add(i.__getattr__(attribute))
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

def plot(data : List[Tuple[DictionaryPlus,str]], autorange = True, directory=os.getcwd()+'/interactive_plots.html'):
    
    p = bopl.figure(height=800, width=1400, tools=["box_zoom", 'reset', 'wheel_zoom'],
                    x_axis_type="datetime", x_axis_location="above",
                    background_fill_color="#efefef")

    def get_type(d : DictionaryPlus) ->str:
        if 'CO(ppm)' in d.show().columns:
            return 'CO(ppm)'
        elif 'pm_adj' in d.show().columns:
            return 'pm_adj'
        elif 'PM2_5MC' in d.show().columns:
            return 'PM2_5MC'
    def get_max(d : DictionaryPlus) -> float:
        t = get_type(d)
        return max([max(value[t]) for value in d.values()])

    def get_min(d: DictionaryPlus) -> float:
        t = get_type(d)
        return min([min(value[t]) for value in d.values()])
    
    ranges= {'PM2_5MC': 'upas', 'CO(ppm)': 'lascar', 'pm_adj': 'purpleAir'}
    colors = itertools.cycle(palette)

    if autorange:
        y_range_left = (min([get_min(item[0]) for item in data if item[1] == 'left']), max(
            [get_max(item[0]) for item in data if item[1] == 'left']))
        y_range_right = None if 'right' not in [i[1] for i in data] else (min([get_min(item[0]) for item in data if item[1] == 'right']), max([get_max(item[0]) for item in data if item[1] == 'right']))
        autorange = (y_range_left,y_range_right)

    p.y_range = Range1d(autorange[0][0], autorange[0][1])
    if 'right' in [i[1] for i in data]:
        p.extra_y_ranges['las'] = Range1d(autorange[1][0], autorange[1][1])
    y_range_label = 'PM2.5 [μg/m3]' if (get_type([i[0] for i in data if i[1]=='left'][0]) == 'pm_adj') | (
        get_type([i[0] for i in data if i[1] == 'left'][0]) == 'PM2_5MC') else 'CO [ppm]'
    if 'right' in [i[1] for i in data]:
        y_range2_label = 'PM2.5 [μg/m3]' if (get_type([i[0] for i in data if i[1] == 'right'][0]) == 'pm_adj') | (get_type([i[0] for i in data if i[1] == 'right'][0]) == 'PM2_5MC') else 'CO [ppm]'


    select = bopl.figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                        tools="", toolbar_location=None, background_fill_color="#efefef")

    p.yaxis.axis_label = y_range_label
    p.yaxis.axis_label_orientation='vertical'
    p.yaxis.axis_label_text_font_size='10px'

    if 'right' in [i[1] for i in data]:
        ax2 = LinearAxis(y_range_name="las",
                        axis_label=y_range2_label, axis_label_orientation=math.radians(90), axis_label_text_font_size='10px')
        p.add_layout(ax2, 'right')

    range_tool_x = RangeTool(x_range=p.x_range)
    range_tool_x.overlay.fill_color = "navy"
    range_tool_x.overlay.fill_alpha = 0.2


    for item in data:
        color = next(colors)
        if item[1] == 'left':
            for tt in set_attrib(item[0],'type'):
                for ll in set_attrib(subset(item[0],{'type':[tt]}),'location'):
                    for j in range(len(subset(item[0],{'type':[tt],'location':[ll]}))):
                        b = show(subset(item[0],{'type':[tt],'location':[ll]}), j)
                        dates = np.array(b.index, dtype=np.datetime64)
                        source = ColumnDataSource(data=dict(date=dates, close=b[get_type(item[0])]))
                        x = p.line('date', 'close', source=source, alpha=0.7,
                                   muted_alpha=0.05, legend_label=' '.join([ranges[get_type(item[0])], b.meta['type'].split(' ')[0], b.meta['location']]), color=color)
                        select.line('date', 'close', source=source )
                        select.ygrid.grid_line_color = None
                        select.add_tools(range_tool_x)

        else:
            for tt in set_attrib(item[0], 'type'):
                for ll in set_attrib(subset(item[0], {'type': [tt]}), 'location'):
                    for j in range(len(subset(item[0], {'type': [tt], 'location': [ll]}))):
                        b = show(
                            subset(item[0], {'type': [tt], 'location': [ll]}), j)
                        dates = np.array(b.index, dtype=np.datetime64)
                        source = ColumnDataSource(
                            data=dict(date=dates, close=b[get_type(item[0])]))
                        x = p.line('date', 'close', source=source, muted_alpha=0.05, y_range_name="las", color=color,
                                alpha=0.7, legend_label=' '.join([ranges[get_type(item[0])], b.meta['type'].split(' ')[0], b.meta['location']]))


    p.add_layout(p.legend[0], 'right')

    p.legend.click_policy = "mute"
    #bopl.show(column(p, select))
    bopl.save(column(p, select),filename=directory)