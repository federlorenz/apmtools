import pandas as pd
import numpy as np
import copy


class DictionaryPlus(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return DictionaryPlus

    def show(self, number=0):
        """
        return an element of a dictionary
        If number is not specified, returns the values associated with the first key
        """
        try:
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

        return DictionaryPlus(a)
    

    def set_attrib(self, attribute):
        """
        returns the set of attribute values for dictionary
        """
        return_set = set()
        for i in self.values():
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
    
    def metadata(self, listall = False):
        meta = set().union(
            *[set(i.meta.keys()) for i in self.values()])
        if listall:
            return {key:self.set_attrib(key) for key in meta}
        else: 
            return meta

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
        self.metadata = {}
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
