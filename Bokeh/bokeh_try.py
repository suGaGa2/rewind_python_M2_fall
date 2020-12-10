from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, ImageURL, Plot, Range1d
from bokeh.plotting import figure, output_file, show, Column, Row
from bokeh.io import curdoc, save

from bokeh.models.widgets import Button
from bokeh.themes import built_in_themes
import pandas as pd

df_sx   = pd.read_csv("../CSVs/S_X_output.csv")
sx_value_list = df_sx["sx_value"].values.tolist()
index_list = df_sx["i"].values.tolist()
start_datetime_list = df_sx["start_datetime"].values.tolist()
end_datetime_list   = df_sx["end_datetime"].values.tolist()

source_sx = ColumnDataSource(
    data = {'index': index_list, 'sx_value': sx_value_list,\
            'start_datetime' : start_datetime_list, 'end_datetime' : end_datetime_list}
)
TOOLTIPS = [
    ("index",          "@index"),
    ("sx_value",       "@sx_value"),
    ("start_datetime", "@start_datetime"),
    ("end_datetime",   "@end_datetime")
]

curdoc().theme = 'night_sky'
p_sx =  figure(plot_width=960, plot_height=540,tooltips=TOOLTIPS,
               title="Significanse Curve")        #p = <class 'bokeh.plotting.figure.Figure'>

p_sx.line(  x='index', y='sx_value', source=source_sx, line_width=2)
p_sx.circle(x='index', y='sx_value', source=source_sx)
#----------------------------------------------------------------------------------------#
# Thumbnail Crowdの表示
df_crd = pd.read_csv("../CSVs/after_forced_output.csv")
p_c_x_list = df_crd["p_c_x"].values.tolist()
p_c_y_list = df_crd["p_c_y"].values.tolist()
rect_width_list
rect_height_list
color_list

p.rect(x=[1, 2, 3], y=[1, 2, 3], width=0.2, height=40, color="#CAB2D6",
       angle=pi/3, height_units="screen")


source_sx = ColumnDataSource(
p_c_x_list = 
    data = {'p_c_x': index_list, 'p_c_y': sx_value_list,\
            'start_datetime' : start_datetime_list, 'end_datetime' : end_datetime_list}
)

p_cr = figure(plot_width=960, plot_height=540,tooltips=TOOLTIPS,
               title="Significanse Curve")        #p = <class 'bokeh.plotting.figure.Figure'>

#最後の部分
output_file("MOGE.html")
curdoc().add_root(p_sx)
show(p_sx)


        

'''
plot = figure(plot_width=400, plot_height=400) #p = <class 'bokeh.plotting.figure.Figure'>



source = ColumnDataSource(
    data = {'x': [1, 5, 9], 'y': [1, 5, 9], 'url' : ['1.png', '2.png', '1.png']}
)

#renderer = p.scatter(x='x', y='y', source=source, color='color', size=10)
#r1 = plot.circle(x='x', y='y', source=source, color='red', size=10)
#r1 = <class 'bokeh.models.renderers.GlyphRenderer'>
r2 = plot.image_url(url='url' ,x='x', y='y', w=0.8, h=0.6,source=source, anchor="center")
#image1 = ImageURL(url=['1.png'], x="x", y="y", w=0.6, h=0.6, anchor="center", global_alpha = 0.5)
#r2 = plot.add_glyph(source, image1)
print(type(r2))
#r2 = <class 'bokeh.models.renderers.GlyphRenderer'>




#r2 = p.image_url(url=['2.png'], x=1, y=3, w=0.8, h=0.6)

draw_tool = PointDrawTool(renderers=[r2])
plot.add_tools(draw_tool)
plot.toolbar.active_tap = draw_tool

columns = [TableColumn(field="x", title="x"),
           TableColumn(field="y", title="y"),
           TableColumn(field='color', title='url')]
table = DataTable(source=source, columns=columns, editable=True, height=200)

button_area = Column(sepal_button, petal_button, table)
layout = Row(button_area, plot)
output_file("MOGE.html")
show(layout)
#curdoc().add_root(layout)

'''

