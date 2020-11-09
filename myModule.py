import datetime
import numpy as np

import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sip

import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import math
import copy

from PIL import Image, ImageDraw
import requests
import os

import MeCab



class Watch:
    def __init__(self, video_title, video_id, channel_name, channel_id, watch_datetime):
        self.video_title = video_title
        self.video_id = video_id
        self.channel_name = channel_name
        self.channel_id = channel_id
        self.watch_datetime = datetime.datetime.strptime(str(watch_datetime), '%b %d, %Y, %I:%M:%S %p JST')
        self.tags = []


class Watchs:
    def __init__(self, path):
        print(path)
        self.df = pd.read_csv(path)
        self.df = self.df.dropna(how ="any")
        self.df = self.df[~self.df['watch_date'].str.contains('Wa')]
        self.df = self.df[~self.df['watch_date'].str.contains('Wt')]

        self.watch_list_all = []
        self.watch_list_selected = []
        self.watch_start_datetime_all  = 0
        self.watch_end_datetime_all    = 0

    def construct_watch_list_all(self):
        row_no = len(self.df)
        i = 0
        while i < row_no:
            watch = Watch(self.df.iat[i, 0], self.df.iat[i, 1], self.df.iat[i, 2], self.df.iat[i, 3], self.df.iat[i, 4])  
            self.watch_list_all.append(watch)
            i += 1

        #　watchsの start_datetime と end_datetime を抽出
        self.watch_start_datetime_all = self.watch_list_all[0].watch_datetime
        self.watch_end_datetime_all = self.watch_list_all[-1].watch_datetime
        
    def set_watch_list_selected(self, start_datetime, end_datetime):
        self.watch_list_selected.clear()
        for watch in self.watch_list_all:
            if watch.watch_datetime <= start_datetime and watch.watch_datetime >= end_datetime:
                self.watch_list_selected.append(watch)

    def tag_each_watch(self):
        for watch in self.watch_list_all:
            mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd').parse(watch.video_title)
            lines = mecab.split('\n')

            nounAndVerb = []
            for line in lines:
                feature = line.split('\t')
                if len(feature) == 2:#'EOS'と''を省く
                    info = feature[1].split(',')
                    hinshi = info[0]
                    if hinshi in ('名詞', '動詞'):
                        nounAndVerb.append(info[6])

            watch.tags = nounAndVerb
            while '*' in watch.tags:
                watch.tags.remove('*')


'''
-------------------------------------------------------------------------------
'''
class TelementWInfo:
    def __init__(self):
        self.frequency = 0
        self.position = np.zeros(2)
        self.watch_video_dict = {} # {video_id : [watch_1, watch_2, watch_3], video_id : [watch_1, watch_2], ....}
        self.color = "NO"
        self.histgram_position = [0, 0, 0, 0]

    
    def set_histgram_position(self, which, interval_num=0, max=0, min=0):
        if which == 0: #frequency
            interval = (max - min) / interval_num
            i = 0
            while self.frequency >= min + interval * (i + 1):
                i += 1
            self.histgram_position[which] = i
        
        if which == 1: #x position
            interval = (max - min) / interval_num
            i = 0
            while self.position[0] >= min + interval * (i + 1):
                i += 1
            self.histgram_position[which] = i
        
        if which == 2: #x position
            interval = (max - min) / interval_num
            i = 0
            while self.position[1] >= min + interval * (i + 1):
                i += 1
            self.histgram_position[which] = i

        if which == 3: #color
            if self.color == "RED":
                self.histgram_position[which] = 0
            if self.color == "BLUE":
                self.histgram_position[which] = 1
            if self.color == "PURPLE":
                self.histgram_position[which] = 2
            if self.color == "NO":
                self.histgram_position[which] = 3



class Telement:
    def __init__(self,start_datetime, end_datetime, i):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.word_count_dict = {}    #{channelid : frequency,  channelid : frequency,} ソートされて、上位の選ばれたもの
        self.extracted_w_info_dict = {} # {channelid : w_info, id, w_info}
        self.index = i
        self.all_histgram_position_dict = {} # {[1,2,4,0]:1, [3,6,7,8]:5,  ...}


    def max_frequency(self):
        max = 0
        i = 0
        for w_info in self.extracted_w_info_dict.values():
            if i == 0:
                max = w_info.frequency
            elif max < w_info.frequency:
                max = w_info.frequency
            i += 1
        return max

    def min_frequency(self):
        min = 0
        i = 0
        for w_info in self.extracted_w_info_dict.values():
            if i == 0:
                min = w_info.frequency
            elif min > w_info.frequency:
                min = w_info.frequency
            i += 1
        return min

    def max_x(self):
        max = 0
        for w_info in self.extracted_w_info_dict.values():
            if w_info.position[0] > max:
                max = w_info.position[0]
        return max

    def min_x(self):
        min = 0
        for (i, w_info) in zip( range(len(self.extracted_w_info_dict.values())), self.extracted_w_info_dict.values() ):
            if i == 0:
                min = w_info.position[0]
            elif w_info.position[0] < min:
                min = w_info.position[0]
        return min

    def max_y(self):
        max = 0
        for w_info in self.extracted_w_info_dict.values():
            if w_info.position[1] > max:
                max = w_info.position[1]
        return max

    def min_y(self):
        min = 0
        for (i, w_info) in zip( range(len(self.extracted_w_info_dict.values())), self.extracted_w_info_dict.values()):
            if i == 0:
                min = w_info.position[1]
            elif w_info.position[1] < min:
                min = w_info.position[1]
        return min




class Tset:
    def __init__(self, start_datetime, end_datetime):
        self.interval_num = 0
        self.start_datetime = start_datetime
        self.end_datetime   = end_datetime
        self.interval_time = 0
        self.elements_list = []
        self.x_max = 0
        self.y_max = 0
        self.x_min = 0
        self.y_min = 0

    def set_interval_num(self, interval_num):
        self.interval_num = interval_num
        self.elements_list.clear()
        self.interval_time = (self.start_datetime - self.end_datetime) / self.interval_num
    
    # t_setを、t_elementに、start_datetime, end_datetime, index(0オリジン)を登録して生成。
    def construct_element_list(self):
        tmp = self.start_datetime
        for i in range(self.interval_num):
            if i != self.interval_num - 1: #最後以外。境界条件の関係で
                t_element = Telement(tmp, tmp - self.interval_time, i)
                self.elements_list.append(t_element)
                tmp -= self.interval_time
                i += 1
            if i == self.interval_num -1: #最後だったら。境界条件的にこれいる。
                t_element = Telement(tmp, tmp - self.interval_time*1.1, i) #なんとなく1.1
                self.elements_list.append(t_element)

    def construct_t_element_word_count_dict_2nd(self, watchs, WORDS_NUM_IN_A_CLOUD):
        # t_elemnt.word_dictを作る。一つの期間で、出現回数まとめる。
        for t_element in self.elements_list:
            for watch in watchs.watch_list_all:
                if watch.watch_datetime  <= t_element.start_datetime and watch.watch_datetime >= t_element.end_datetime:
                    for tag in watch.tags:
                        if tag not in t_element.word_count_dict: 
                            t_element.word_count_dict[tag] = 1
                        else:
                            t_element.word_count_dict[tag] += 1
                
            tmp_taple_list = sorted(t_element.word_count_dict.items(), key=lambda x:x[1], reverse=True) #多い順に並び替え。返り値がタプルのリストになっているため、辞書に置き換える必要がある。
            tmp_taple_list = tmp_taple_list[0:WORDS_NUM_IN_A_CLOUD] #出現回数が閾値以降のものを残す.
            #タプルを辞書に直して格納
            t_element.word_count_dict.clear()
            for item in tmp_taple_list:
                t_element.word_count_dict[item[0]] = item[1]
            #print(t_element.word_count_dict)

    def construct_t_element_word_count_dict(self, watchs, WORDS_NUM_IN_A_CLOUD):
        # t_elemnt.word_dictを作る。一つの期間で、出現回数まとめる。
        for t_element in self.elements_list:
            for watch in watchs.watch_list_all:
                if watch.watch_datetime  <= t_element.start_datetime and watch.watch_datetime >= t_element.end_datetime:
                    if watch.channel_id not in t_element.word_count_dict: #channel_idで作る。
                        t_element.word_count_dict[watch.channel_id] = 1
                    else:
                        t_element.word_count_dict[watch.channel_id] += 1
                
            tmp_taple_list = sorted(t_element.word_count_dict.items(), key=lambda x:x[1], reverse=True) #多い順に並び替え。返り値がタプルのリストになっているため、辞書に置き換える必要がある。
            tmp_taple_list = tmp_taple_list[0:WORDS_NUM_IN_A_CLOUD] #出現回数が閾値以降のものを残す.
            #タプルを辞書に直して格納
            t_element.word_count_dict.clear()
            for item in tmp_taple_list:
                t_element.word_count_dict[item[0]] = item[1]
            #print(t_element.word_count_dict)

    def set_t_element_extracted_w_info_dic(self, watchs, w_set):
        # t_elementの extracted_w_info_dictに情報を挿入
        for (i, t_element) in zip(range(len(self.elements_list)), self.elements_list):
            # watch_video_dictを作るために、
            watchs.set_watch_list_selected(t_element.start_datetime, t_element.end_datetime)
            for channel_id in t_element.word_count_dict.keys():
                t_element.extracted_w_info_dict[channel_id] = TelementWInfo()
                # frequency ポインタじゃないから、実体を直接変える必要がある。
                self.elements_list[i].extracted_w_info_dict[channel_id].frequency = t_element.word_count_dict[channel_id]
                
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

    def draw_crowd(self, DRAW_INDEX):
        X_SIZE = 500
        Y_SIZE = 500

        campus = Image.new('RGB', (X_SIZE, Y_SIZE), (128, 128, 128))
        position_scale_rate = 500 / (self.x_max + 0.5) 
        draw = ImageDraw.Draw(campus)
        # 一回 DRAW_INDEX　のt_elementに対して、描画してみる。
        for channel_id in self.elements_list[DRAW_INDEX].extracted_w_info_dict:
            x = self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].position[0]
            y = self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].position[1]
            size = self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].frequency

            if self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].color == "RED":
                draw.rectangle((X_SIZE / 2 + position_scale_rate * x -2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y -2 * size, \
                                X_SIZE / 2+ position_scale_rate * x +2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y +2 * size), \
                                fill=(240, 0, 0), outline=(255, 255, 255))

            if self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].color == "BLUE":
                draw.rectangle((X_SIZE / 2 + position_scale_rate * x -2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y -2 * size, \
                                X_SIZE / 2 + position_scale_rate * x +2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y +2 * size), \
                                fill=(0, 0, 240), outline=(255, 255, 255))

            if self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].color == "PURPLE":
                draw.rectangle((X_SIZE / 2 + position_scale_rate * x -2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y -2 * size, \
                                X_SIZE / 2 + position_scale_rate * x +2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y +2 * size), \
                                fill=(150, 0, 150), outline=(255, 255, 255))
                
            if self.elements_list[DRAW_INDEX].extracted_w_info_dict[channel_id].color == "NO":
                draw.rectangle((X_SIZE / 2 + position_scale_rate * x -2 * size, \
                                X_SIZE / 2 + position_scale_rate * y -2 * size, \
                                Y_SIZE / 2 + position_scale_rate * x +2 * size, \
                                Y_SIZE / 2 + position_scale_rate * y +2 * size), \
                                fill=(50, 50, 50), outline=(255, 255, 255))

        campus.save('pillow_imagedraw.jpg', quality=95)

    def draw_significance_curve(self, TSET_INTERVAL_NUM):
        # significance curve 書く
        left   = np.array( range(TSET_INTERVAL_NUM-1) )
        height = np.empty(0)
        for i in range(TSET_INTERVAL_NUM-1):
            # H(X)を求めるために、t[5]に対して、ヒストグラムを作成する。
            S_INDEX = i
            INTERVAL_NUM = 5  #論文だと64になっていたやつ
            for w_info in self.elements_list[S_INDEX].extracted_w_info_dict.values():
                # frequency
                w_info.set_histgram_position(0, INTERVAL_NUM, self.elements_list[S_INDEX].max_frequency(), self.elements_list[S_INDEX].min_frequency())
                # x position
                w_info.set_histgram_position(1, INTERVAL_NUM, self.elements_list[S_INDEX].max_x()        , self.elements_list[S_INDEX].min_x()        )
                # y position
                w_info.set_histgram_position(2, INTERVAL_NUM, self.elements_list[S_INDEX].max_y()        , self.elements_list[S_INDEX].min_y()        )
                # color 
                w_info.set_histgram_position(3, INTERVAL_NUM)

                if str(w_info.histgram_position) not in self.elements_list[S_INDEX].all_histgram_position_dict:
                    self.elements_list[S_INDEX].all_histgram_position_dict[str(w_info.histgram_position)] =  1
                else:
                    self.elements_list[S_INDEX].all_histgram_position_dict[str(w_info.histgram_position)] += 1

            # H(X)を計算
            all_sum = sum(self.elements_list[S_INDEX].all_histgram_position_dict.values())
            H_X = 0
            for cnt in self.elements_list[S_INDEX].all_histgram_position_dict.values():
                p = cnt / all_sum
                H_X += p * math.log( 1 / p)

            # H(X;Y)を計算
            n = 1
            for w_info in self.elements_list[S_INDEX + n].extracted_w_info_dict.values():
                # frequency
                w_info.set_histgram_position(0, INTERVAL_NUM, self.elements_list[S_INDEX + n].max_frequency(), self.elements_list[S_INDEX + n].min_frequency())
                # x position
                w_info.set_histgram_position(1, INTERVAL_NUM, self.elements_list[S_INDEX + n].max_x()        , self.elements_list[S_INDEX + n].min_x()        )
                # y position
                w_info.set_histgram_position(2, INTERVAL_NUM, self.elements_list[S_INDEX + n].max_y()        , self.elements_list[S_INDEX + n].min_y()        )
                # color 
                w_info.set_histgram_position(3, INTERVAL_NUM)

            # joint histgramを作る
            # S_INDEXに対して
            j_hist_dict = {} # {channel_id: [[1,3,4,2][0,1,3,4]],  channel_id: [[-1][0,1,3,4]]       }
            for item in self.elements_list[S_INDEX].extracted_w_info_dict.items():
                myid = item[0]
                histgram_position = item[1].histgram_position
                if myid not in j_hist_dict:
                    j_hist_dict[myid] = [histgram_position, [-1]]

            # S_INDEX + n に対して
            for item in self.elements_list[S_INDEX + n].extracted_w_info_dict.items():
                myid = item[0]
                histgram_position = item[1].histgram_position
                if myid not in j_hist_dict:
                    j_hist_dict[myid] = [[-1], histgram_position]
                else:
                    j_hist_dict[myid][1] = histgram_position

            j_hist_count_dict = {}
            for position in j_hist_dict.values():
                if str(position) not in j_hist_count_dict:
                    j_hist_count_dict[str(position)] = 1
                else:
                    j_hist_count_dict[str(position)] += 1


            #print(j_hist_count_dict)
            all_sum = sum(j_hist_count_dict.values())
            H_X_semicolon_Y = 0
            for item in j_hist_count_dict.items():
                position_str = item[0]
                cnt_xy = item[1]
                p_xy = cnt_xy / all_sum

                #p_x計算する
                x_str = '[' + position_str.split(", [")[0][1:]
                cnt_x = 0
                for key in j_hist_count_dict.keys():
                    if x_str in key:
                        cnt_x += 1
                p_x = cnt_x / all_sum
                
                #p_y計算する
                y_str = '[' + position_str.split(", [")[1][:-1]
                cnt_y = 0
                for key in j_hist_count_dict.keys():
                    if y_str in key:
                        cnt_y += 1
                p_y = cnt_y / all_sum

                H_X_semicolon_Y += p_xy * math.log( p_xy / (p_x * p_y))

            S_X = H_X - H_X_semicolon_Y + 1.4
            height = np.append(height, S_X)
        print(len(height))
        #文字を取り込む
        #fig = plt.figure()
        #p = plt.vlines([DRAW_INDEX], -1, 2.1, "red", linestyles='dashed') 
        #plt.plot(left, height)
        #fig.savefig("S_X.png") 

'''
-------------------------------------------------------------------------------
'''
class Welement:
    def __init__(self, interval_num):
        self.importance_vec = np.zeros(interval_num)
        self.position = np.zeros(2)

class Wset:    #選び抜かれた、Wordクラウドに出てくるchannel idの集まり
    def __init__(self):
        self.elements_dict = {} #{channelid: w_element, channelid: w_element, } w_elementには、importance_vec と　positionが含まれる。
    
    def set_elements_dict(self, t_set):
        self.elements_dict.clear()
        for t_element in t_set.elements_list:
            for channel_id in t_element.word_count_dict.keys():
                if not channel_id in self.elements_dict:
                    self.elements_dict[channel_id] = Welement(t_set.interval_num)

    def set_element_dict_importance_vec(self, t_set):
        # w_elementのimportance Vecを設定
        for i, t_element in zip(range(t_set.interval_num), t_set.elements_list):
            for item in t_element.word_count_dict.items():
                self.elements_dict[item[0]].importance_vec[i] = item[1]

    def set_words_initital_position(self,t_set):
        # Initial Positionを決めるために、 マトリックスを作成
        size = len(self.elements_dict)
        matrix = np.empty((size, size))
        for i, w_element_1 in zip(range(size), self.elements_dict.values()):
            for j, w_element_2 in zip(range(size), self.elements_dict.values()):
                vec_1 = w_element_1.importance_vec
                vec_2 = w_element_2.importance_vec

                vec_1 = vec_1 / np.linalg.norm(vec_1, ord=2) 
                vec_2 = vec_2 / np.linalg.norm(vec_2, ord=2) 

                value = np.dot(vec_1, vec_2)
                matrix[i][j] = 1 - value
        mds = MDS(n_components=2, dissimilarity="precomputed")
        X_2d = mds.fit_transform(matrix)
        t_set.x_max = np.max(X_2d, axis = 0)[0]
        t_set.y_max = np.max(X_2d, axis = 0)[1]
        t_set.x_min = np.min(X_2d, axis = 0)[0]
        t_set.y_min = np.min(X_2d, axis = 0)[1]
        # w element に position を入れる。この時に、x, yの最大値、最小値も調べておく。
        for (x, y), w_element in zip(X_2d, self.elements_dict.values()):
            w_element.position = np.array([x, y])
        
    
    def print_w_set(self):
        for w_element in self.elements_dict.items():
            print(w_element)
            print("****")
        


class MainWindow(QWidget):
    def __init__(self, image1, image2, start_datetime, end_datetime, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('Trend View')

        self.vbox  = QVBoxLayout()
        self.vbox2 = QVBoxLayout()
        self.parent_hbox = QHBoxLayout()

        # QPixmapオブジェクト作成
        self.pixmap1 = QPixmap(image1)
        self.pixmap2 = QPixmap(image2)

        # ラベルを作ってその中に画像を置く
        self.lbl1 = QLabel()
        self.lbl1.setPixmap(self.pixmap1)
        self.lbl_time = QLabel()
        self.lbl_time.setText("     " + end_datetime.strftime('%Y/%m/%d') + " ~ " + start_datetime.strftime('%Y/%m/%d'))

        self.lbl2 = QLabel()
        self.lbl2.setPixmap(self.pixmap2)

        self.vbox.addWidget(self.lbl1)
        self.vbox.addWidget(self.lbl_time)
        
        self.button = QPushButton('change')
        self.button.clicked.connect(self.change)

        self.inputText = QLineEdit()
        self.inputText.setText("")

        self.vbox2.addWidget(self.lbl2)
        self.vbox2.addWidget(self.inputText)
        self.vbox2.addWidget(self.button)

        self.parent_hbox.addLayout(self.vbox)
        self.parent_hbox.addLayout(self.vbox2)
        self.setLayout(self.parent_hbox)
        self.move(300, 200)
        self.show()  
    
    def change(self):
        draw_index = int(self.inputText.text())
        start_datetime, end_datetime = make_figure(DRAW_INDEX=draw_index)
        self.pixmap1 = QPixmap("pillow_imagedraw.jpg")
        self.lbl1.setPixmap(self.pixmap1)
        self.pixmap2 = QPixmap("S_X.png")
        self.lbl2.setPixmap(self.pixmap2)
        self.lbl_time.setText("              " + end_datetime.strftime('%Y/%m/%d') + " ~ " + start_datetime.strftime('%Y/%m/%d'))
'''
'''
        

        

'''
-------------------------------------------------------------------------------
'''