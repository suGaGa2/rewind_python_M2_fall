from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, ImageURL, Plot, Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show, Column, Row
from bokeh.io import curdoc, save

from bokeh.models.widgets import Button, Slider, Toggle
from bokeh.themes import built_in_themes
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime as dt

from PIL import Image, ImageDraw, ImageFont
from PillowRoundedRecCreation import word_image_creation


'''
def slider_onchange(attr, old, new):
    a = slider.value
    image_tcrd = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", global_alpha=slider.value,  anchor="center")
    r_tcrd_2 = p_crd.add_glyph(source_tcrd, image_tcrd)
'''

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

p_sx.line(  x='start_datetime_dt', y='sx_value', source=source_sx, line_width=4, color='#08F7FE')
p_sx.circle(x='start_datetime_dt', y='sx_value', source=source_sx)
#----------------------------------------------------------------------------------------#
# Thumbnail Crowdの表示
df_crd = pd.read_csv("Bokeh/CSVs/afrer_forced_output.csv")
#===========================================================================
df_wcrd = df_crd[df_crd["color"] != "Thumbnail"]
p_c_x_list_wcrd = df_wcrd["p_c_x"].values.tolist()
p_c_y_list_wcrd = df_wcrd["p_c_y"].values.tolist()

word_list_wcrd = df_wcrd["word"].values.tolist()
url_list_wcrd = list(map(lambda x: "Bokeh/static/IMAGEs/" + x + ".png", word_list_wcrd))

p_br_x_list_wcrd = df_wcrd["p_br_x"].values.tolist()
p_bl_x_list_wcrd = df_wcrd["p_bl_x"].values.tolist()
w_list_wcrd = (np.array(p_br_x_list_wcrd) - np.array(p_bl_x_list_wcrd)).tolist()

p_bl_y_list_wcrd = df_wcrd["p_bl_y"].values.tolist()
p_tl_y_list_wcrd = df_wcrd["p_tl_y"].values.tolist()
h_list_wcrd = (np.array(p_bl_y_list_wcrd) - np.array(p_tl_y_list_wcrd)).tolist()

alpha_list_wcrd = np.ones(len(df_wcrd))
# **************************************************************************************
df_tcrd  = df_crd[df_crd["color"] == "Thumbnail"]
p_c_x_list_tcrd = df_tcrd["p_c_x"].values.tolist()
p_c_y_list_tcrd = df_tcrd["p_c_y"].values.tolist()

word_list_tcrd = df_tcrd["word"].values.tolist()
url_list_tcrd = list(map(lambda x: "Bokeh/static/IMAGEs/" + x + ".png", word_list_tcrd))

p_br_x_list_tcrd = df_tcrd["p_br_x"].values.tolist()
p_bl_x_list_tcrd = df_tcrd["p_bl_x"].values.tolist()
w_list_tcrd = (np.array(p_br_x_list_tcrd) - np.array(p_bl_x_list_tcrd)).tolist()

p_bl_y_list_tcrd = df_tcrd["p_bl_y"].values.tolist()
p_tl_y_list_tcrd = df_tcrd["p_tl_y"].values.tolist()
h_list_tcrd = (np.array(p_bl_y_list_tcrd) - np.array(p_tl_y_list_tcrd)).tolist()

alpha_list_tcrd = np.ones(len(df_tcrd))
#=======================================================
'''
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
'''
###***************************************************************************
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


source_wcrd = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_wcrd, 'p_c_y': p_c_y_list_wcrd,\
            'url_list': url_list_wcrd,   'w'    : w_list_wcrd,   'h' : h_list_wcrd, 'alpha' : alpha_list_wcrd}
)

source_tcrd = ColumnDataSource(
    data = {'p_c_x': p_c_x_list_tcrd, 'p_c_y': p_c_y_list_tcrd,\
            'url_list': url_list_tcrd, 'w' : w_list_tcrd, 'h' : h_list_tcrd, 'alpha' : alpha_list_tcrd}
)


p_crd =  figure(plot_width=900, plot_height=540,
               title="Crowd", match_aspect=True, tools='pan,wheel_zoom,box_zoom,undo,redo,save,reset,help') 

r_wcrd_1 = p_crd.circle(x='p_c_x', y='p_c_y', source=source_wcrd, color='red', alpha=0, size=10)
image_wcrd = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", anchor="center")
r_wcrd_2 = p_crd.add_glyph(source_wcrd, image_wcrd)

r_tcrd_1 = p_crd.circle(x='p_c_x', y='p_c_y', source=source_tcrd, color='red', alpha=0, size=10)
image_tcrd = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", anchor="center")
r_tcrd_2 = p_crd.add_glyph(source_tcrd, image_tcrd)

draw_tool = PointDrawTool(renderers=[r_wcrd_1, r_wcrd_2, r_tcrd_1, r_tcrd_2])
p_crd.add_tools(draw_tool)
p_crd.toolbar.active_tap = draw_tool

'''
btn_wcrd = Button(label="Word")
btn_wcrd.on_click(btn_wcrd_onclick)
btn_tcrd = Button(label="Thumbnail", button_type="success")
btn_tcrd.on_click(btn_tcrd_onclick)
btn_wt_crd = Button(label="Word & Thumbnail",  button_type="success")
btn_wt_crd.on_click(btn_wt_crd_onclick)
'''
toggle1 = Toggle(label="Word", button_type="success", active=True)
toggle1.js_link('active', r_wcrd_2, 'visible')

toggle2 = Toggle(label="Thumbnail", button_type="success", active=True)
toggle2.js_link('active', r_tcrd_2, 'visible')

thubmnail_alpha_slider = Slider(start=0, end=10, value=1, step=.1, title="Stuff")
slider = Slider(start=0, end=1, value=1, step=.1, title="Alpha Value")
#slider.on_change('value', slider_onchange)
slider.js_link('value', image_tcrd, 'global_alpha')

toggles = Row(toggle1, toggle2)
controles=Column(toggles, slider)

columns = [
        TableColumn(field="p_c_x", title="p_c_x"),
        TableColumn(field="p_c_y", title="p_c_y")
          ]
data_table = DataTable(source=source_tcrd, columns=columns, width=400, height=280)


p_crd_layout = Column(p_crd, controles, data_table)

#最後の部分
plots = Row(p_sx, p_crd_layout)
#output_file("MOGE.html")
show(plots)
curdoc().add_root(plots)




