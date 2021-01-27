from bokeh.models import DataTable, TableColumn, PointDrawTool,BoxSelectTool, ColumnDataSource, ImageURL,Plot, Range1d, DatetimeTickFormatter,  Circle, Slope, BoxAnnotation, CustomJS, Band, CDSView,
from bokeh.plotting import figure, output_file, show, Column, Row
from bokeh.io import curdoc, save

from bokeh.models.widgets import Button, Slider, Toggle
from bokeh.themes import built_in_themes
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
from PillowRoundedRecCreation import word_image_creation


TSET_INTERVAL_NUM = 45
INDEX = 14
DATA_PATH = THUMBNAIL_CROWD_PATH = "Bokeh/CSVs/afrer_forced_output_" + str(INDEX) + '.csv'


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

# SOURCE_SX
source_sx = ColumnDataSource(
    data = {'index': index_list, 'sx_value': sx_value_list,\
            'start_datetime_str' : start_datetime_list_str, 'end_datetime_str' : end_datetime_list_str,\
            'start_datetime_dt' : start_datetime_list_dt,   'end_datetime_dt'  : end_datetime_list_dt}
)


source_sx_persistency = ColumnDataSource(
    data = {'index': [], 'sx_value': [],\
            'start_datetime_str' : [], 'end_datetime_str':[],\
            'start_datetime_dt'  : [],  'end_datetime_dt' :[]}
)

TOOLTIPS = [
    ("index",          "@index"),
    ("sx_value",       "@sx_value"),
    ("start_datetime", "@start_datetime_str"),
    ("end_datetime",   "@end_datetime_str")
]
#**************************************************************
curdoc().theme = 'night_sky'
#**************************************************************
p_sx =  figure(plot_width=600, plot_height=450,
               tools='pan,wheel_zoom, box_zoom, box_select, crosshair, undo,redo,save,reset,help',
               tooltips=TOOLTIPS,
               title="Significanse Curve", x_axis_type='datetime',
               background_fill_color = "#282C44",
               background_fill_alpha = 1)       #p = <class 'bokeh.plotting.figure.Figure'>
p_sx.yaxis.minor_tick_in = 5


renderer_line = p_sx.line(  x='start_datetime_dt', y='sx_value', source=source_sx,
                            line_width=2, color='#08F7FE',
                            line_alpha=0.7,
                            selection_alpha=1,
                            nonselection_alpha=1,
)
band = Band(base='start_datetime_dt', lower=0, upper='sx_value', source=source_sx, level='underlay',
            fill_color='#08F7FE', fill_alpha=0.1, line_width=0, line_color='black')
p_sx.add_layout(band)
#*******************************************************************
renderer_circle = p_sx.circle(x='start_datetime_dt', y='sx_value',
                              source=source_sx, size = 7,
                              fill_color='#08F7FE', fill_alpha=0.3,
                              line_color='#08F7FE', line_width=2)

selected_circle = Circle(fill_alpha = 1, fill_color='#FE53BB')
nonselected_circle = Circle(fill_alpha=0.2, fill_color='#08F7FE', line_color="blue")
renderer_circle.selection_glyph = selected_circle
renderer_circle.nonselection_glyph = nonselected_circle
select_overlay = p_sx.select_one(BoxSelectTool).overlay
select_overlay.fill_color = "firebrick"
select_overlay.line_color = None
# ***********************************************:
box = BoxAnnotation(bottom=10, fill_alpha=0.1, fill_color='red')
p_sx.add_layout(box)


p_sx.circle(x='start_datetime_dt', y='sx_value',
             source=source_sx_persistency, size=10,
             fill_color='fuchsia', fill_alpha=0.5,
             line_color='crimson', line_width=1)



#----------------------------------------------------------------------------------------#
# Thumbnail Crowdの表示
INDEX = 0
DATA_PATH = "Bokeh/CSVs/afrer_forced_output_" + str(INDEX) + '.csv'
df_crd = pd.read_csv(DATA_PATH)
df_crd['draw_index'] = INDEX

INDEX += 1
while INDEX < TSET_INTERVAL_NUM:
    DATA_PATH = "Bokeh/CSVs/afrer_forced_output_" + str(INDEX) + '.csv'
    df_will_appended = pd.read_csv(DATA_PATH)
    df_will_appended['draw_index'] = INDEX
    df_crd = pd.concat([df_crd, df_will_appended)
    INDEX += 1

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

index_list_wcrd = df_wcrd["index"].values.tolist()

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

index_list_tcrd = df_wcrd["index"].values.tolist()

alpha_list_tcrd = np.ones(len(df_tcrd))


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
            'url_list': url_list_wcrd,   'w'    : w_list_wcrd,   'h' : h_list_wcrd, 'alpha' : alpha_list_wcrd, 'index' : index_list_wcrd}
)
booleans = [ True if index == 2 else False for index in source_wcrd.data['index'] ]

source_tcrd = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_tcrd, 'p_c_y': p_c_y_list_tcrd,\
            'url_list': url_list_tcrd,   'w'    : w_list_tcrd,   'h' : h_list_tcrd, 'alpha' : alpha_list_tcrd, 'index' : index_list_tcrd}
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


slider_sx = Slider(start=0, end=5, value=5, step=.1, title="Persistency")
slider_sx.js_on_change('value',
                      CustomJS(args=dict(box=box, source_sx=source_sx, source_sx_persistency=source_sx_persistency),
    code="""
    box.bottom= cb_obj.value;
    var d1 = source_sx.data;
    const d2 = {'index': [], 'sx_value': [],
                'start_datetime_str': [], 'end_datetime_str' : [], 
                'start_datetime_dt':  [], 'end_datetime_dt'  : []};
    for (var i = 0; i < d1['index'].length; i++) {
        if (d1['sx_value'][i] > cb_obj.value) {
                d2['index'].push(             d1['index'][i]);
                d2['sx_value'].push(          d1['sx_value'][i]);
                d2['start_datetime_str'].push(d1['start_datetime_str'][i]);
                d2['end_datetime_str'].push(  d1['end_datetime_str'][i]);
                d2['start_datetime_dt'].push( d1['start_datetime_dt'][i]);
                d2['end_datetime_dt'].push(   d1['end_datetime_dt'][i]);
            }

    }
    source_sx_persistency.data = d2;
    console.log(d2);
    source_sx_persistency.change.emit();
    """))



toggle_sx = Toggle(label="Persistence", button_type="success", active=True)
toggle1 = Toggle(label="Word", button_type="success", active=True)
toggle1.js_link('active', r_wcrd_2, 'visible')

toggle2 = Toggle(label="Thumbnail", button_type="success", active=True)
toggle2.js_link('active', r_tcrd_2, 'visible')

slider = Slider(start=0, end=1, value=1, step=.1, title="Alpha Value")
#slider.on_change('value', slider_onchange)
slider.js_link('value', image_tcrd, 'global_alpha')

toggles = Row(toggle1, toggle2)
controles=Column(toggles, slider)


columns = [
        TableColumn(field="start_datetime_dt", title="start_datetime_dt"),
        TableColumn(field="sx_value", title="sx_value")
          ]
data_table = DataTable(source=source_sx_persistency, columns=columns, width=400, height=280)

p_sx_layout  = Column(p_sx, slider_sx, controles)
p_crd_layout = Column(p_crd, data_table)

#最後の部分
plots = Row(p_sx_layout, p_crd_layout)
#output_file("MOGE.html")
show(plots)
curdoc().add_root(plots)




