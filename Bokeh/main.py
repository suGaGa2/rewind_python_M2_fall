from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, ImageURL, Plot, Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show, Column, Row
from bokeh.io import curdoc, save

from bokeh.models.widgets import Button
from bokeh.themes import built_in_themes
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime as dt

from PIL import Image, ImageDraw, ImageFont
from PillowRoundedRecCreation import word_image_creation


def btn_wcrd_onclick():
    df_wcrd = df_crd[df_crd["color"] != "Thumbnail"]
    
    source_crd = ColumnDataSource(
    data = {'p_c_x': p_c_x_list, 'p_c_y': p_c_y_list,\
            'url_list': url_list, 'w' : w_list, 'h' : h_list}
    )
    


#df_sx   = pd.read_csv("../CSVs/S_X_output.csv")
df_sx   = pd.read_csv("Bokeh/CSVs/S_X_output.csv")
sx_value_list = df_sx["sx_value"].values.tolist()
index_list = df_sx["i"].values.tolist()
start_datetime_list = df_sx["start_datetime"].values.tolist()
start_datetime_list_str = list(map(lambda x: x.split(".")[0], start_datetime_list))
start_datetime_list_dt = list(map(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S') , start_datetime_list_str))

end_datetime_list   = df_sx["end_datetime"].values.tolist()
end_datetime_list_str = list(map(lambda x: x.split(".")[0], end_datetime_list))
end_datetime_list_dt = list(map(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S') , end_datetime_list_str))

print(type(start_datetime_list[1]))
source_sx = ColumnDataSource(
    data = {'index': index_list, 'sx_value': sx_value_list,\
            'start_datetime_str' : start_datetime_list_str, 'end_datetime_str' : end_datetime_list_str,\
            'start_datetime_dt' : start_datetime_list_dt,   'end_datetime_dt'  : end_datetime_list_dt}
)
TOOLTIPS = [
    ("index",          "@index"),
    ("sx_value",       "@sx_value"),
    ("start_datetime", "@start_datetime_str"),
    ("end_datetime",   "@end_datetime_str")
]

curdoc().theme = 'night_sky'
p_sx =  figure(plot_width=600, plot_height=450,tooltips=TOOLTIPS,
               title="Significanse Curve", x_axis_type='datetime')        #p = <class 'bokeh.plotting.figure.Figure'>

p_sx.line(  x='start_datetime_dt', y='sx_value', source=source_sx, line_width=2)
p_sx.circle(x='start_datetime_dt', y='sx_value', source=source_sx)
#----------------------------------------------------------------------------------------#
# Thumbnail Crowdの表示
#df_crd = pd.read_csv("../CSVs/afrer_forced_output.csv")
df_crd = pd.read_csv("Bokeh/CSVs/afrer_forced_output.csv")
p_c_x_list = df_crd["p_c_x"].values.tolist()
p_c_y_list = df_crd["p_c_y"].values.tolist()

word_list = df_crd["word"].values.tolist()
url_list = list(map(lambda x: "Bokeh/static/IMAGEs/" + x + ".png", word_list))
#url_list  = list(map(lambda x: "./IMAGES/" + x + ".png", word_list))

p_br_x_list = df_crd["p_br_x"].values.tolist()
p_bl_x_list = df_crd["p_bl_x"].values.tolist()
w_list = (np.array(p_br_x_list) - np.array(p_bl_x_list)).tolist()

p_bl_y_list = df_crd["p_bl_y"].values.tolist()
p_tl_y_list = df_crd["p_tl_y"].values.tolist()
h_list = (np.array(p_bl_y_list) - np.array(p_tl_y_list)).tolist()


for index, row in df_crd.iterrows():
    if not os.path.exists('Bokeh/static/IMAGEs/' + row['word']  + ".png"):
        if row['color'] != 'Thumbnail':
            word_image_creation(row['word'], row['size'], row['color'], "Bokeh/static/IMAGEs")
        if row['color'] == 'Thumbnail':
            url = 'http://i.ytimg.com/vi/' + row['word'] + "/mqdefault.jpg"
            response = requests.get(url)
            image = response.content
            file_name = "Bokeh/static/IMAGEs/" + row['word']  + ".png"
            with open(file_name, "wb") as aaa:
                aaa.write(image)
            img = Image.open("Bokeh/static/IMAGEs/" + row['word']  + ".png")
            img_resize = img.resize( (int((row['p_br_x'] - row['p_bl_x']) / 2), int((row['p_bl_y'] - row['p_tl_y']) / 2)) )

source_crd = ColumnDataSource(
    data = {'p_c_x': p_c_x_list, 'p_c_y': p_c_y_list,\
            'url_list': url_list, 'w' : w_list, 'h' : h_list}
)

p_crd =  figure(plot_width=900, plot_height=540,
               title="Crowd", match_aspect=True) 

r1 = p_crd.circle(x='p_c_x', y='p_c_y', source=source_crd, color='red', alpha=1, size=10)
image1 = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", anchor="center")
r2 = p_crd.add_glyph(source_crd, image1)

draw_tool = PointDrawTool(renderers=[r1, r2])
p_crd.add_tools(draw_tool)
p_crd.toolbar.active_tap = draw_tool


btn_wcrd = Button(label="Word",  button_type="success")
btn_wcrd.on_click(btn_wcrd_onclick)
btn_twrd = Button(label="Thumbnail", button_type="success")
btn_twrd.on_click(btn_twrd_onclick)
btn_wt_crd = Button(label="Word & Thumbnail",  button_type="success")
btn_wt_crd.on_click(btn_wt_crd_onclick)

buttons = Row(btn_crd, btn_thmb, btn_crd_and_thmb)

p_crd_layout = Column(p_crd, buttons)

#最後の部分
plots = Row(p_sx, p_crd_layout)
#output_file("MOGE.html")
show(plots)
curdoc().add_root(plots)




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

