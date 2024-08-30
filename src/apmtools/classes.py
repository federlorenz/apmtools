import pandas as pd
import numpy as np
import copy


import itertools
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.layouts import column, layout
import bokeh.plotting as bopl
import os as os

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
            if key!=None:
                return (self.subset({self.filter_key:[key]}).show())
            else:
                if type(number) == type(""):
                    return (self.subset({self.filter_key: [number]}).show())
                else:
                    return (self[list(self.keys())[number]])
        except:
            print("something's wrong")

    def subset(self, filter_dict, filter_style='all'):
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
        return_dict = copy.deepcopy(self)
        if filter_style=='any':
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
                                if getattr(value,'meta')[i] in j:
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
        if filter_style == 'all':
            a = {key:value for key,value in return_dict.items()}         
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
                    elif hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                        try:
                            if type(j) == type(""):
                                if not eval("value.__getattr__('meta')[\""+i+"\"]" + j):
                                    del a[key]
                                    break
                            else:
                                if getattr(value, 'meta')[i] not in j:
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
                                if getattr(value,i) in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    elif hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                        try:
                            if type(j) == type(""):
                                if eval("value.__getattr__('meta')[\""+i+"\"]" + j):
                                    del a[key]
                                    break
                            else:
                                if getattr(value,'meta')[i] in j:
                                    del a[key]
                                    break
                        except:
                            pass
                    else:
                        break


        a = DictionaryPlus(a)
        a.filter_key = self.filter_key
        return a
    

    def set_attrib(self, attribute):
        """
        returns the set of attribute values for dictionary
        """
        return_set = set()
        for i in self.values():
            if hasattr(i, 'meta') & (type(i.meta) == type({})) & (attribute in i.meta.keys()):
                try:
                    return_set.add(getattr(i,'meta')[attribute])
                except:
                    pass
            else:
                try:
                    return_set.add(getattr(i,attribute))
                except:
                    pass

        return return_set
    
    def metadata(self, listall = False):
        meta = set().union(
            *[set(i.meta.keys()) for i in self.values()])
        if listall:
            return {key:self.set_attrib(key) for key in meta}
        else: 
            return meta
        
    def apply_func(self,func):
        a = DictionaryPlus({key: func(value) for key, value in self.items()})
        a.filter_key = self.filter_key
        return a

class Apm(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.meta = {}
    _metadata = ['meta']

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
            self = self.loc[self.index <= date_end]
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

class ApmSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        pd.Series.__init__(self, *args, **kwargs)
    _metadata = ['meta']

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
            self = self.loc[self.index <= date_end]
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
    def __init__(self, *args, **kwargs):
        Apm.__init__(self, *args, **kwargs)
        self.meta = {}
    _metadata = ['meta']

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
    _metadata = ['meta']

    @property
    def _constructor(self):
        return SumSeries

class PolarH10(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self["ecg"] = None
        self["acc"] = None
        self["rr"] = None
        self["hr"] = None
        self.meta = {}
    _metadata = ['meta']

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
    def __init__(self, figures=1, shape=column, titles=None, figure_sizes=(800, 1400)):
        self.all_figures = []
        for j in range((figures)):
            self.all_figures.append(bopl.figure(height=figure_sizes[0], width=figure_sizes[1], tools=["box_zoom", 'reset', 'wheel_zoom', "pan"],
                                                x_axis_type="datetime", x_axis_location="above",
                                                background_fill_color="#efefef"))
            shape(self.all_figures)
        if titles == None:
            for j in range(len(self.all_figures)):
                self.all_figures[j].title.text = f"Figure {j}"
                self.all_figures[j].title.align = "center"
                self.all_figures[j].title.text_font_size = "25px"
        else:
            for j in range(len(self.all_figures)):
                self.all_figures[j].title.text = titles[j]
                self.all_figures[j].title.align = "center"
                self.all_figures[j].title.text_font_size = "25px"
        self.colors = itertools.cycle(palette)
        self.range_start = None
        self.range_end = None

    def add_data(self, datain: DictionaryPlus, variable, plotn, filterdict=None, label="", color=True):
        datain = datain.subset(filterdict) if filterdict != None else datain
        if color:
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
        if len(datain) == 0:
            pass
        else:
            for value in datain.values():
                dates = np.array(value.index, dtype=np.datetime64)
                source = ColumnDataSource(
                    data=dict(date=dates, close=value[variable]))
                x = self.all_figures[plotn].line('date', 'close', source=source, alpha=0.7,
                                                 muted_alpha=0.05, legend_label=label, color=color)

    def finalize(self, axis_labels=False, plot_layout=None):
        datarange = DataRange1d(start=self.range_start-(self.range_end-self.range_start)/20,
                                end=self.range_end+(self.range_end-self.range_start)/20)
        for j in range(len(self.all_figures)):
            self.all_figures[j].add_layout(
                self.all_figures[j].legend[0], 'right')
            self.all_figures[j].legend.click_policy = "mute"
        self.all_figures[0].x_range = datarange
        for j in range(1, len(self.all_figures)):
            self.all_figures[j].x_range = self.all_figures[0].x_range
        if axis_labels:
            for j in range(len(self.all_figures)):
                self.all_figures[j].yaxis.axis_label = axis_labels[j]
                self.all_figures[j].yaxis.axis_label_orientation = 'vertical'
                self.all_figures[j].yaxis.axis_label_text_font_size = '10px'
        if plot_layout == None:
            self.layout = column(self.all_figures)
        else:
            self.layout = layout(plot_layout)

    def show(self):
        bopl.show(self.layout)

    def save(self, filename=os.getcwd()+'/interactive_plots.html'):
        bopl.save(self.layout, filename=filename)
