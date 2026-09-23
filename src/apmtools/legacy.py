
import itertools
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.layouts import column, layout
import bokeh.plotting as bopl
from bokeh.models.axes import DatetimeAxis, MercatorAxis
from .classes import DictionaryPlus
import xyzservices.providers as xyz
import numpy as np
import os as os

class BPlot():
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
                source = ColumnDataSource(
                    data=dict(date=dates, close=value[variable]))
                if plotn == None:
                    x = self.all_figures[-1].line('date', 'close', source=source, alpha=0.7,
                                                  muted_alpha=0.05, legend_label=label, color=color)
                else:
                    x = self.all_figures[plotn].line('date', 'close', source=source, alpha=0.7,
                                                     muted_alpha=0.05, legend_label=label, color=color)

    def add_data_vertical(self, datain: DictionaryPlus, variable, range_variable, plotn=None, filterdict=None, label="", color=None, line_width=0.3):
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

            maximus = max([value[range_variable].max()
                          for value in datain.values()])
            minimum = 0 if (0 < min([value[range_variable].min() for value in datain.values(
            )])) else min([value[range_variable].min() for value in datain.values()])

            for value in datain.values():
                if plotn == None:
                    for s in value[variable].value_counts().index:
                        left = value.loc[value[variable] == s].index[0]
                        right = value.loc[value[variable] == s].index[-1]
                        x = self.all_figures[-1].quad(left=left, right=right, top=maximus, bottom=minimum, alpha=0.02,
                                                      muted_alpha=0.2, legend_label=label, fill_color=color, line_alpha=0)
                else:
                    for s in value[variable].value_counts().index:
                        left = value.loc[value[variable] == s].index[0]
                        right = value.loc[value[variable] == s].index[-1]
                        x = self.all_figures[plotn].quad(left=left, right=right, top=maximus, bottom=minimum, alpha=0.02,
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
