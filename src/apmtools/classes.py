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
