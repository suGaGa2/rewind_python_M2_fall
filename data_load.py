import pandas as pd
import numpy as np
import datetime
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import math

from myModule import Watch, Watchs, TelementWInfo, Telement, Tset, Welement, Wset

from PIL import Image, ImageDraw
import requests
import os

df = pd.read_csv("output.csv")
df = df.dropna(how ="any")

df = df[~df['watch_date'].str.contains('Wa')]
df = df[~df['watch_date'].str.contains('Wt')]


row_no = len(df)


watchs = Watchs()
i = 0
while i < row_no:
    watch = Watch(df.iat[i, 0], df.iat[i, 1], df.iat[i, 2], df.iat[i, 3], df.iat[i, 4])  
    watchs.watch_list_all.append(watch)
    i += 1

#　watchsの start_datetime と end_datetime を抽出
watchs.set_watch_start_datetime_all()
watchs.set_watch_end_datetime_all()

# t_setを作成・初期化
t_set = Tset(watchs.watch_start_datetime_all, watchs.watch_end_datetime_all)
t_set.set_interval_num(20

) #!!!!!!!!!!!!!!!!!!!!!  ここでインターバル数を決める。最低でも3以上にしてください！！     ！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!

# t_setを、t_elementに、start_datetime, end_datetime, index(0オリジン)を登録して生成。　
tmp = t_set.start_datetime
for i in range(t_set.interval_num):
    if i != t_set.interval_num - 1: #最後以外。境界条件の関係で
        t_element = Telement(tmp, tmp - t_set.interval_time, i)
        t_set.add_elements(t_element)
        tmp -= t_set.interval_time
        i += 1
    if i == t_set.interval_num -1: #最後だったら。境界条件的にこれいる。
        t_element = Telement(tmp, tmp - t_set.interval_time*1.1, i) #なんとなく1.1
        t_set.add_elements(t_element)

#出現回数が閾値以降のものを残す時に、！！！！！！！！！ここで何個wordを抽出するかを決める値。ここをいじって！   !!!!!!!!!!!!!!!!!!!!!!!!!
WORDS_NUM_IN_A_CLOUD = 15
# t_elemnt.channel_dictを作る。一つの期間で、出現回数まとめる。
for t_element in t_set.elements_list:
    for watch in watchs.watch_list_all:
        if watch.watch_datetime  <= t_element.start_datetime and watch.watch_datetime >= t_element.end_datetime:
            if watch.channel_id not in t_element.channel_count_dict: #channel_idで作る。
                t_element.channel_count_dict[watch.channel_id] = 1
            else:
                t_element.channel_count_dict[watch.channel_id] += 1
        
    tmp_taple_list = sorted(t_element.channel_count_dict.items(), key=lambda x:x[1], reverse=True) #多い順に並び替え。返り値がタプルのリストになっているため、辞書に置き換える必要がある。
    tmp_taple_list = tmp_taple_list[0:WORDS_NUM_IN_A_CLOUD] #出現回数が閾値以降のものを残す.
    #タプルを辞書に直して格納
    t_element.channel_count_dict.clear()
    for item in tmp_taple_list:
        t_element.channel_count_dict[item[0]] = item[1]
    #print(t_element.channel_count_dict)

# w_set作る。初期情報登録
w_set = Wset()
w_set.set_elements_dict(t_set)

# w_elementのimportance Vecを設定
for i, t_element in zip(range(t_set.interval_num), t_set.elements_list):
    for item in t_element.channel_count_dict.items():
        w_set.elements_dict[item[0]].importance_vec[i] = item[1]


# Initial Positionを決めるために、 マトリックスを作成
size = len(w_set.elements_dict)

matrix = np.empty((size, size))

for i, w_element_1 in zip(range(size), w_set.elements_dict.values()):
    for j, w_element_2 in zip(range(size), w_set.elements_dict.values()):
        vec_1 = w_element_1.importance_vec
        vec_2 = w_element_2.importance_vec

        vec_1 = vec_1 / np.linalg.norm(vec_1, ord=2) 
        vec_2 = vec_2 / np.linalg.norm(vec_2, ord=2) 

        value = np.dot(vec_1, vec_2)
        
        matrix[i][j] = 1 - value


#print(matrix)
mds = MDS(n_components=2, dissimilarity="precomputed")
X_2d = mds.fit_transform(matrix)
t_set.x_max = np.max(X_2d, axis = 0)[0]
t_set.y_max = np.max(X_2d, axis = 0)[1]
t_set.x_min = np.min(X_2d, axis = 0)[0]
t_set.y_min = np.min(X_2d, axis = 0)[1]



# w element に position を入れる。この時に、x, yの最大値、最小値も調べておく。
for (x, y), w_element in zip(X_2d, w_set.elements_dict.values()):
    w_element.position = np.array([x, y])

# t_elementの extracted_w_info_dictに情報を挿入

for (i, t_element) in zip(range(len(t_set.elements_list)), t_set.elements_list):
    # watch_video_dictを作るために、
    watchs.set_watch_list_selected(t_element.start_datetime, t_element.end_datetime)
    for channel_id in t_element.channel_count_dict.keys():
        t_element.extracted_w_info_dict[channel_id] = TelementWInfo()
        # frequency ポインタじゃないから、実体を直接変える必要がある。
        t_set.elements_list[i].extracted_w_info_dict[channel_id].frequency = t_element.channel_count_dict[channel_id]
        
        # position
        t_element.extracted_w_info_dict[channel_id].position = w_set.elements_dict[channel_id].position
        
        
        # watch_video_dict
        for watch in watchs.watch_list_selected:
            if channel_id == watch.channel_id:
                if watch.video_id not in t_element.extracted_w_info_dict[channel_id].watch_video_dict:
                    t_element.extracted_w_info_dict[channel_id].watch_video_dict[watch.video_id] = [watch]
                else:
                    t_element.extracted_w_info_dict[channel_id].watch_video_dict[watch.video_id].append(watch)
        
        # color 
        vec = w_set.elements_dict[channel_id].importance_vec # 使い回すから、代入
        index = t_element.index                              # 使い回すから、代入

        if t_element.index == 0: #一番端のT_elementの時
            if vec[index + 1] > 0:
                t_element.extracted_w_info_dict[channel_id].color = "RED"
            elif np.count_nonzero(vec[index + 1:] > 0) == 0:
                t_element.extracted_w_info_dict[channel_id].color = "PURPLE"

        elif t_element.index == len(vec) -1:#一番端のT_elementの時
            if vec[index - 1] > 0:
                t_element.extracted_w_info_dict[channel_id].color = "BLUE"
            elif np.count_nonzero(vec[:index] > 0) == 0:
                t_element.extracted_w_info_dict[channel_id].color = "PURPLE"
        
        else:
            if vec[index + 1] > 0 and vec[index - 1] == 0:
                t_element.extracted_w_info_dict[channel_id].color = "RED"
            elif vec[index - 1] > 0 and vec[index + 1] == 0:
                t_element.extracted_w_info_dict[channel_id].color = "BLUE"
            elif np.count_nonzero(vec[index + 1:] > 0) == 0 and np.count_nonzero(vec[:index] > 0) == 0:
                t_element.extracted_w_info_dict[channel_id].color = "PURPLE"
        #print(channel_id, ": color =  ",  t_element.extracted_w_info_dict[channel_id].color)
        #print("---------------------------")



# significance curve 書く
# H(X)を求めるために、t[5]に対して、ヒストグラムを作成する。
S_INDEX = 5
INTERVAL_NUM = 5  #論文だと64になっていたやつ
all_histgram_position_dict = {}
for w_info in t_set.elements_list[S_INDEX].extracted_w_info_dict.values():
    # frequency
    w_info.set_histgram_position(0, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_frequency(), t_set.elements_list[S_INDEX].min_frequency())
    # x position
    w_info.set_histgram_position(1, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_x()        , t_set.elements_list[S_INDEX].min_x()        )
    # y position
    w_info.set_histgram_position(2, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_y()        , t_set.elements_list[S_INDEX].min_y()        )
    # color 
    w_info.set_histgram_position(3, INTERVAL_NUM)

    if str(w_info.histgram_position) not in all_histgram_position_dict:
        all_histgram_position_dict[str(w_info.histgram_position)] =  1
    else:
        all_histgram_position_dict[str(w_info.histgram_position)] += 1

# H(X)を計算
all_sum = sum(all_histgram_position_dict.values())
H_X = 0
for cnt in all_histgram_position_dict.values():
    p = cnt / all_sum
    H_X += p * math.log( 1 / p)
print(H_X)

# H(X;Y)を計算
for w_info in t_set.elements_list[S_INDEX + 1].extracted_w_info_dict.values():
    # frequency
    w_info.set_histgram_position(0, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_frequency(), t_set.elements_list[S_INDEX].min_frequency())
    # x position
    w_info.set_histgram_position(1, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_x()        , t_set.elements_list[S_INDEX].min_x()        )
    # y position
    w_info.set_histgram_position(2, INTERVAL_NUM, t_set.elements_list[S_INDEX].max_y()        , t_set.elements_list[S_INDEX].min_y()        )
    # color 
    w_info.set_histgram_position(3, INTERVAL_NUM)




    




'''
campus = Image.new('RGB', (1000, 1000), (128, 128, 128))
position_scale_rate = 500 / (t_set.x_max + 0.5) 
draw = ImageDraw.Draw(campus)

# 一回 index 5　のt_elementに対して、描画してみる。

for channel_id in t_set.elements_list[5].extracted_w_info_dict:
    x = t_set.elements_list[5].extracted_w_info_dict[channel_id].position[0]
    y = t_set.elements_list[5].extracted_w_info_dict[channel_id].position[1]
    size = t_set.elements_list[5].extracted_w_info_dict[channel_id].frequency

    if t_set.elements_list[5].extracted_w_info_dict[channel_id].color == "RED":
        draw.rectangle((500 + position_scale_rate * x -2 * size, \
                        500 + position_scale_rate * y -2 * size, \
                        500 + position_scale_rate * x +2 * size, \
                        500 + position_scale_rate * y +2 * size), \
                        fill=(240, 0, 0), outline=(255, 255, 255))

    if t_set.elements_list[5].extracted_w_info_dict[channel_id].color == "BLUE":
        draw.rectangle((500 + position_scale_rate * x -2 * size, \
                        500 + position_scale_rate * y -2 * size, \
                        500 + position_scale_rate * x +2 * size, \
                        500 + position_scale_rate * y +2 * size), \
                        fill=(0, 0, 240), outline=(255, 255, 255))

    if t_set.elements_list[5].extracted_w_info_dict[channel_id].color == "PURPLE":
        draw.rectangle((500 + position_scale_rate * x -2 * size, \
                        500 + position_scale_rate * y -2 * size, \
                        500 + position_scale_rate * x +2 * size, \
                        500 + position_scale_rate * y +2 * size), \
                        fill=(150, 0, 150), outline=(255, 255, 255))
        
    if t_set.elements_list[5].extracted_w_info_dict[channel_id].color == "NO":
        draw.rectangle((500 + position_scale_rate * x -2 * size, \
                        500 + position_scale_rate * y -2 * size, \
                        500 + position_scale_rate * x +2 * size, \
                        500 + position_scale_rate * y +2 * size), \
                        fill=(50, 50, 50), outline=(255, 255, 255))

campus.save('pillow_imagedraw.jpg', quality=95)
'''




