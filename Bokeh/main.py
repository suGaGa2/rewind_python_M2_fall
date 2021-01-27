from bokeh.models import DataTable, TableColumn, PointDrawTool,BoxSelectTool, ColumnDataSource, ImageURL,Plot, Range1d, DatetimeTickFormatter,  Circle, Slope, BoxAnnotation, CustomJS, Band, CDSView, Spinner
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
FIRST_DRAW_INDEX  = 0 #初期に描画する時間窓のINDEX
#INDEX = 14
#DATA_PATH = THUMBNAIL_CROWD_PATH = "Bokeh/CSVs/afrer_forced_output_" + str(INDEX) + '.csv'




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

source_sx_draw_highlight = ColumnDataSource(
    data = {'index': [index_list[FIRST_DRAW_INDEX]], 'sx_value': [sx_value_list[FIRST_DRAW_INDEX]],\
            'start_datetime_str' : [start_datetime_list_str[FIRST_DRAW_INDEX]], 'end_datetime_str':[end_datetime_list_str[FIRST_DRAW_INDEX]],\
            'start_datetime_dt'  : [start_datetime_list_dt[FIRST_DRAW_INDEX]],  'end_datetime_dt' :[end_datetime_list_dt[FIRST_DRAW_INDEX]]
           }
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


p_sx.circle(x='start_datetime_dt', y='sx_value',
             source=source_sx_draw_highlight, size=20,
             fill_color='coral', fill_alpha=0.5,
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
    df_crd = pd.concat([df_crd, df_will_appended])
    INDEX += 1
    print("processing INDEX " + str(INDEX))

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

draw_index_list_wcrd = df_wcrd["draw_index"].values.tolist()

extract_start_index_wcrd      = draw_index_list_wcrd.index(FIRST_DRAW_INDEX)  # FIRST_DRAW_INDEXがはじまる位置．はじめに描画する p_crd の時間窓のINDEXを得るため
extract_element_num_wcrd      = draw_index_list_wcrd.count(FIRST_DRAW_INDEX)  # FIRST_DRAW_INDEXの個数．はじめに描画する p_crd の時間窓のINDEXを得るため
extract_end_index_wcrd        = extract_start_index_wcrd + extract_element_num_wcrd - 1 # FIRST_DRAW_INDEXがおわる位置．

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

draw_index_list_tcrd = df_tcrd["draw_index"].values.tolist()
extract_start_index_tcrd      = draw_index_list_tcrd.index(FIRST_DRAW_INDEX)  # FIRST_DRAW_INDEXがはじまる位置．はじめに描画する p_crd の時間窓のINDEXを得るため
extract_element_num_tcrd      = draw_index_list_tcrd.count(FIRST_DRAW_INDEX)  # FIRST_DRAW_INDEXの個数．はじめに描画する p_crd の時間窓のINDEXを得るため
extract_end_index_tcrd       = extract_start_index_tcrd + extract_element_num_tcrd - 1 # FIRST_DRAW_INDEXがおわる位置．

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




# *************************************************

# 【ワード用】全 Draw Index の情報が詰まった ColumnDataSource
source_wcrd = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_wcrd, 'p_c_y': p_c_y_list_wcrd,\
            'url_list': url_list_wcrd,   'w'    : w_list_wcrd,   'h' : h_list_wcrd,
            'alpha' : alpha_list_wcrd, 'draw_index' : draw_index_list_wcrd}
)

# 【ワード用】描画するようの ColumnDataSource
source_wcrd_view = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_wcrd[extract_start_index_wcrd:(extract_end_index_wcrd+1)],
            'p_c_y': p_c_y_list_wcrd[extract_start_index_wcrd   :(extract_end_index_wcrd+1)],
            'url_list': url_list_wcrd[extract_start_index_wcrd   :(extract_end_index_wcrd+1)],
            'w'    : w_list_wcrd[extract_start_index_wcrd       :(extract_end_index_wcrd+1)],
            'h' : h_list_wcrd[extract_start_index_wcrd          :(extract_end_index_wcrd+1)],
            'alpha' : alpha_list_wcrd[extract_start_index_wcrd  :(extract_end_index_wcrd+1)],
            'draw_index' : draw_index_list_wcrd[extract_start_index_wcrd  :(extract_end_index_wcrd+1)]
            }
)


# ***************************************************

# 【サムネイル用】全 Draw Index の情報が詰まった ColumnDataSource
source_tcrd = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_tcrd, 'p_c_y': p_c_y_list_tcrd,\
            'url_list': url_list_tcrd,   'w'    : w_list_tcrd,   'h' : h_list_tcrd, 'alpha' : alpha_list_tcrd, 'draw_index' : draw_index_list_tcrd}
)

# 【サムネイル用】描画するようの ColumnDataSource
source_tcrd_view = ColumnDataSource(
    data = {'p_c_x'   : p_c_x_list_tcrd[extract_start_index_tcrd:(extract_end_index_tcrd+1)],
            'p_c_y': p_c_y_list_tcrd[extract_start_index_tcrd   :(extract_end_index_tcrd+1)],
            'url_list': url_list_tcrd[extract_start_index_tcrd   :(extract_end_index_tcrd+1)],
            'w'    : w_list_tcrd[extract_start_index_tcrd       :(extract_end_index_tcrd+1)],
            'h' : h_list_tcrd[extract_start_index_tcrd          :(extract_end_index_tcrd+1)],
            'alpha' : alpha_list_tcrd[extract_start_index_tcrd  :(extract_end_index_tcrd+1)],
            'draw_index' : draw_index_list_tcrd[extract_start_index_tcrd  :(extract_end_index_tcrd+1)]
            }
)


# ****************************************************


p_crd =  figure(plot_width=900, plot_height=540,
               title="Crowd", match_aspect=True, tools='pan,wheel_zoom,box_zoom,undo,redo,save,reset,help') 

r_wcrd_1 = p_crd.circle(x='p_c_x', y='p_c_y', source=source_wcrd_view, color='red', alpha=0, size=10)
image_wcrd = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", anchor="center")
r_wcrd_2 = p_crd.add_glyph(source_wcrd_view, image_wcrd)

r_tcrd_1 = p_crd.circle(x='p_c_x', y='p_c_y', source=source_tcrd_view, color='red', alpha=0, size=10)
image_tcrd = ImageURL(url="url_list", x="p_c_x", y="p_c_y", w="w", h="h", anchor="center")
r_tcrd_2 = p_crd.add_glyph(source_tcrd_view, image_tcrd)

draw_tool = PointDrawTool(renderers=[r_wcrd_1, r_wcrd_2, r_tcrd_1, r_tcrd_2])
p_crd.add_tools(draw_tool)
p_crd.toolbar.active_tap = draw_tool


slider_sx = Slider(start=0, end=5, value=5, step=.1, title="Persistency", sizing_mode='stretch_width')
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

slider_alpha_Value = Slider(start=0, end=1, value=1, step=.1, title="Alpha Value")
#slider.on_change('value', slider_onchange)
slider_alpha_Value.js_link('value', image_tcrd, 'global_alpha')

silder_draw_index = Slider(start=0, end=TSET_INTERVAL_NUM-1, value=FIRST_DRAW_INDEX, step=-1, title="Draw Index", direction="rtl", sizing_mode='stretch_width')
callback = CustomJS(
        args=dict(
            source_wcrd=source_wcrd, source_wcrd_view=source_wcrd_view,
            source_tcrd=source_tcrd, source_tcrd_view=source_tcrd_view,
            source_sx=source_sx,
            source_sx_draw_highlight=source_sx_draw_highlight),
        code="""
        var d1 = source_wcrd.data;
        var d2 = source_wcrd_view.data;
        var d3 = source_tcrd.data;
        var d4 = source_tcrd_view.data;

        for (var i = 0; i < d2['url_list'].length; i++) {
            for (var j = 0; j < d1['url_list'].length; j++) {
                if ( d2['url_list'][i]== d1['url_list'][j] ) {
                    d1['p_c_x'][j] = d2['p_c_x'][i]
                    d1['p_c_y'][j] = d2['p_c_y'][i]
                    d1['url_list'][j] = d2['url_list'][i]
                    d1['w'][j] = d2['w'][i]
                    d1['h'][j] = d2['h'][i]
                    d1['alpha'][j] = d2['alpha'][i]
                    d1['draw_index'][j] = d2['draw_index'][i]
                }
            }
        }
        for (var i = 0; i < d4['url_list'].length; i++) {
            for (var j = 0; j < d3['url_list'].length; j++) {
                if ( d4['url_list'][i]== d3['url_list'][j] ) {
                    d3['p_c_x'][j] = d4['p_c_x'][i]
                    d3['p_c_y'][j] = d4['p_c_y'][i]
                    d3['url_list'][j] = d4['url_list'][i]
                    d3['w'][j] = d4['w'][i]
                    d3['h'][j] = d4['h'][i]
                    d3['alpha'][j] = d4['alpha'][i]
                    d3['draw_index'][j] = d4['draw_index'][i]
                }
            }
        }


        var draw_index = cb_obj.value;
        d2['p_c_x'] = []
        d2['p_c_y'] = []
        d2['url_list'] = []
        d2['w'] = []
        d2['h'] = []
        d2['alpha'] = []
        d2['draw_index'] = []
        
        for (var i = 0; i < d1['draw_index'].length; i++) {
            if (draw_index == d1['draw_index'][i] ) {
                d2['p_c_x'].push( d1['p_c_x'][i] )
                d2['p_c_y'].push( d1['p_c_y'][i] )
                d2['url_list'].push( d1['url_list'][i] )
                d2['w'].push( d1['w'][i] )
                d2['h'].push( d1['h'][i] )
                d2['alpha'].push( d1['alpha'][i] )
                d2['draw_index'].push( d1['draw_index'][i] )
            }
        }

        source_wcrd_view.change.emit();

        var d3 = source_tcrd.data;
        var d4 = source_tcrd_view.data;
        d4['p_c_x'] = []
        d4['p_c_y'] = []
        d4['url_list'] = []
        d4['w'] = []
        d4['h'] = []
        d4['alpha'] = []
        d4['draw_index'] = []

        for (var i = 0; i < d3['draw_index'].length; i++) {
            if (draw_index == d3['draw_index'][i] ) {
                d4['p_c_x'].push( d3['p_c_x'][i] )
                d4['p_c_y'].push( d3['p_c_y'][i] )
                d4['url_list'].push( d3['url_list'][i] )
                d4['w'].push( d3['w'][i] )
                d4['h'].push( d3['h'][i] )
                d4['alpha'].push( d3['alpha'][i] )
                d4['draw_index'].push( d3['draw_index'][i] )
            }
        }
        source_tcrd_view.change.emit();

        var d6 = source_sx.data;
        const d5 = {'index': [], 'sx_value': [],
                    'start_datetime_str': [], 'end_datetime_str' : [], 
                    'start_datetime_dt':  [], 'end_datetime_dt'  : []};

        d5['index'].push(             d6['index'][draw_index]);
        d5['sx_value'].push(          d6['sx_value'][draw_index]);
        d5['start_datetime_str'].push(d6['start_datetime_str'][draw_index]);
        d5['end_datetime_str'].push(  d6['end_datetime_str'][draw_index]);
        d5['start_datetime_dt'].push( d6['start_datetime_dt'][draw_index]);
        d5['end_datetime_dt'].push(   d6['end_datetime_dt'][draw_index]);

        source_sx_draw_highlight.data = d5;
        source_sx_draw_highlight.change.emit();
        
    """)
silder_draw_index.js_on_change('value', callback)



columns = [
        TableColumn(field="start_datetime_dt", title="start_datetime_dt"),
        TableColumn(field="sx_value", title="sx_value")
          ]
data_table = DataTable(source=source_sx_persistency, columns=columns, width=400, height=280)

controles=Column(silder_draw_index, slider_sx, sizing_mode='stretch_width')
p_sx_layout  = Column(p_sx,  controles, margin=( 8 , 8 , 8 , 8 ))

toggles = Row(toggle1, toggle2)
p_crd_layout = Column(p_crd, toggles, slider_alpha_Value, data_table, margin=( 8 , 8 , 8 , 8 ))

#最後の部分
plots = Row(p_sx_layout, p_crd_layout)
#output_file("MOGE.html")
show(plots)
curdoc().add_root(plots)

