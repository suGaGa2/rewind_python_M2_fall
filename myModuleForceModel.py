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
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import MeCab
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import math
import random 

KR = 10000000
KA = 500000

HOW_MANY_MOVES = 100

class Wrec:
    def __init__(self, word, p_c_x, p_c_y, p_tl_x, p_tl_y, p_tr_x, p_tr_y, p_bl_x, p_bl_y, p_br_x, p_br_y, size,  color, i):
        self.index_num = i
        self.word = word
        self.p_c  = np.array([float(p_c_x),   float(p_c_y )])
        self.p_tl = np.array([float(p_tl_x),  float(p_tl_y)])
        self.p_tr = np.array([float(p_tr_x),  float(p_tr_y)])
        self.p_bl = np.array([float(p_bl_x),  float(p_bl_y)])
        self.p_br = np.array([float(p_br_x),  float(p_br_y)])
        self.size = int(size)
        self.color = color

        self.conneced_wrec_dict = {} #{頂点INDEX：wrec, 頂点INDEX：wrec, ..., 頂点INDEX：wrec}
        self.belonged_mesh =[]        #[[a, b, c], [c, d, e]..., [f, g, h]]
        self.fs = np.zeros(2)
        self.fr = np.zeros(2)
        self.fa = np.zeros(2)
        self.f_all = np.zeros(2)

        self.bfr_p_c = np.zeros(2)
        self.aft_p_c = np.zeros(2)

        self.went_over = False
    
    def calculate_fs(self):
        self.fs = np.zeros(2)
        for wrec in self.conneced_wrec_dict.values():
            d = np.linalg.norm(self.p_c - wrec.p_c)
            one_fs = self.size * wrec.size * d\
                               * ((wrec.p_c - self.p_c)  / np.linalg.norm(wrec.p_c - self.p_c))
            '''
            print("size : " +str(self.size) )
            print("d : " + str(d))
            print((wrec.p_c - self.p_c)  / np.linalg.norm(wrec.p_c - self.p_c))
            print("*********")
            '''
            self.fs += one_fs

    def calculate_fr(self):
        self.fr = np.zeros(2)
        for wrec in self.conneced_wrec_dict.values():
            #重なっている時
            if max(self.p_tl[0], wrec.p_tl[0]) <= min (self.p_br[0], wrec.p_br[0]) and\
               max(self.p_tl[1], wrec.p_tl[1]) <= min (self.p_br[1], wrec.p_br[1]):
               one_fr = KR * min( abs(self.p_br[0]-wrec.p_tl[0]), abs(self.p_br[1]-wrec.p_tl[1]) ) \
                           * ((self.p_c - wrec.p_c) / np.linalg.norm(self.p_c - wrec.p_c))
               self.fr += one_fr
               #print(self.fr)

        def calculate_fa(self):
            a = 1


    def calculate_all_f(self):
        self.f_all = self.fs + self.fr + self.fa



class Wrecset:
    def __init__(self, path):
        self.wrec_list = []
        self.df = pd.read_csv(path)
        row_no = len(self.df)
        i = 0
        while i < row_no:
            wrec = Wrec(self.df.iat[i, 0], self.df.iat[i, 1], self.df.iat[i, 2], self.df.iat[i, 3], \
                        self.df.iat[i, 4], self.df.iat[i, 5], self.df.iat[i, 6], self.df.iat[i, 7],\
                        self.df.iat[i, 8], self.df.iat[i, 9], self.df.iat[i, 10], self.df.iat[i, 11], self.df.iat[i, 12],\
                        i)  
            self.wrec_list.append(wrec)
            i += 1
        self.x_max = 0
        self.x_min = 0
        self.y_max = 0
        self.y_min = 0

        self.word_positions_in_pic = np.zeros(2) #ここは適当に入れてる。

        self.div_value = 0

    def move(self):
        # div_valueを決める
        sum_f_all = 0
        lgt    = len(self.wrec_list)
        for wrec in self.wrec_list:
            sum_f_all += np.linalg.norm( wrec.f_all )
            print("*************************")
            #print(wrec.belonged_mesh)
            print("fs : "    + str(wrec.fs))
            print("fr : "    + str(wrec.fr))
            print("fa : "    + str(wrec.fa))
            #print("f_all : " + str(wrec.f_all))
        sum_f_all = sum_f_all / lgt
        self.div_value = sum_f_all /  1.5
        print("div_value : " + str(self.div_value))

        for wrec in self.wrec_list:
            wrec.bfr_p_c = wrec.p_c
            wrec.p_c     = wrec.p_c  + wrec.f_all / self.div_value #　とりあえず10,000
            wrec.aft_p_c = wrec.p_c
            print("move  : " + str(wrec.f_all  / self.div_value))
            wrec.p_tl = wrec.p_tl + wrec.f_all / self.div_value #　とりあえず10,000
            wrec.p_tr = wrec.p_tr + wrec.f_all / self.div_value #　とりあえず10,000
            wrec.p_bl = wrec.p_bl + wrec.f_all / self.div_value #　とりあえず10,000
            wrec.p_br = wrec.p_br + wrec.f_all / self.div_value #　とりあえず10,000
            
            # print("wrec.f_all : " + str(wrec.f_all))
            #print("bfr_p_c : " + str(wrec.bfr_p_c))
            #print("aft_p_c : " + str(wrec.aft_p_c))
            #print("**************")
        # 移動終了
        # ここでfaを計算しておく。
        for wrec in self.wrec_list:
            # エッジを超えたかを計算して、faを設定していく。
            # ①エッジの直線の方程式 (x1-x2) * (y-y1) - (y1-y2)*(x-x1) = 0　◀︎ 式（1）
            wrec.fa = np.zeros(2) #◀︎一回リセット
            bfr_p_c = wrec.bfr_p_c
            aft_p_c = wrec.aft_p_c

            for mesh in wrec.belonged_mesh: #自分の属する全メッシュに対して
                #print(mesh)
                #print(wrec.index_num)
                tmp_lst = []
                for index in mesh:
                    if index != wrec.index_num:
                        tmp_lst.append(index)
                    
                other_wrecs = np.array(tmp_lst)
                #print("other_wrecs")
                #print(other_wrecs)
                #print("____")

                #print(bfr_p_c)
                #print(wrec.fs)
                #print(aft_p_c)
                p1 = self.wrec_list[other_wrecs[0]].p_c
                p2 = self.wrec_list[other_wrecs[1]].p_c
                s  = (p1[0]- p2[0]) * (bfr_p_c[1]- p1[1]) - (p1[1]- p2[1]) * (bfr_p_c[0] - p1[0]) 
                t  = (p1[0]- p2[0]) * (aft_p_c[1]- p1[1]) - (p1[1]- p2[1]) * (aft_p_c[0] - p1[0]) 
                
                # 点と直線の距離を求める。
                # (y1 - y2)*x + (x2 - x1)*y + x1*y2 - y1*x2 = 0 ◀︎式（１）と同値
                a, b, c = (p1[1]- p2[1]), (p2[0]- p1[0]), (p1[0] *  p2[1] - p1[1]*p2[0])
                x0 , y0 = aft_p_c[0], aft_p_c[1]
                d = abs(a*x0 + b*y0 + c) / math.sqrt(a*a + b*b)
                # 直線 bfr_p_c・aft_p_c 
                # (bfr_p_c[1] - aft_p_c[1]) * x + (aft_p_c[0] - btr_p_c[0]) * y  =   bfr_p_c[1] * aft_p_c[0] - bfr_p_c[0] * aft_p_c[1]
                #
                # エッジの直線
                # (p1[1]      - p2[1]     ) * x + (p2[0]      - p1[0]     ) * y =    p1[1]      * p2[0]      - p1[0]      *  p2[1] 
                A = np.array( [ [bfr_p_c[1] - aft_p_c[1]  ,   aft_p_c[0] - bfr_p_c[0]], \
                                [p1[1]      - p2[1]       ,   p2[0]      - p1[0]     ]  \
                               ])
                #print(A)
                B = np.array( [ bfr_p_c[1] * aft_p_c[0] - bfr_p_c[0] *  aft_p_c[1],\
                                p1[1]      * p2[0]      -  p1[0]     *       p2[1]  \
                              ])
                #print(B)
                try:
                    crossing_point = np.linalg.solve(A, B)
                    fa_direction = crossing_point - aft_p_c

                    if wrec.went_over == False and s * t < 0: # 正常だったのにひっくり返った時
                        wrec.went_over= True 
                        wrec.fa += KA * d * \
                                ( fa_direction / np.linalg.norm(fa_direction))

                    if wrec.went_over == True and s * t < 0: # ひっくり帰っていたのが正常に戻った時
                        wrec.went_over == False

                    if wrec.went_over == True and s * t >= 0: # ひっくり帰ったままの時
                        wrec.fa += KA * d * \
                                ( fa_direction / np.linalg.norm(fa_direction)) 
                except:
                    pass
        
    
    def force_model(self, DRAW_INDEX):
        #　まず同じ値のものはちょっとずらず
        for wrec_1 in self.wrec_list:
            for wrec_2 in self.wrec_list:
                if wrec_1 == wrec_2:
                    continue
                if int(wrec_1.p_c[0]) == int(wrec_2.p_c[0]) and int(wrec_1.p_c[1]) == int(wrec_2.p_c[1]):
                    random_vec = np.array([(0.5 - random.random())*25, (0.5 - random.random())*25])  #とりあえず15なだけ
                    wrec_1.p_c = wrec_1.p_c + random_vec
                    print("come")

        # word_positions_in_pic は wrec_set.wrec_list のINDEX順番と同じ。
        for i, wrec in zip( range(len(self.wrec_list)), self.wrec_list):
            if i == 0:
                word_positions_in_pic = np.array([[wrec.p_c[0], wrec.p_c[1]]])
            if i > 0:
                a_2d_ex = np.array([[wrec.p_c[0], wrec.p_c[1]]])
                word_positions_in_pic = np.append(word_positions_in_pic, a_2d_ex, axis=0)

        #print(word_positions_in_pic)
        tri = Delaunay(word_positions_in_pic)
        fig = delaunay_plot_2d(tri)
        # fig.savefig('./Images/scipy_matplotlib_delaunay_before.png')
        print(tri.simplices)

        for mesh in tri.simplices:
            if mesh[0] == self.wrec_list[mesh[0]].index_num:#  一応一致しているかを確認
                self.wrec_list[mesh[0]].conneced_wrec_dict[mesh[1]] = self.wrec_list[mesh[1]]
                self.wrec_list[mesh[0]].conneced_wrec_dict[mesh[2]] = self.wrec_list[mesh[2]]
                self.wrec_list[mesh[1]].conneced_wrec_dict[mesh[0]] = self.wrec_list[mesh[0]]
                self.wrec_list[mesh[1]].conneced_wrec_dict[mesh[2]] = self.wrec_list[mesh[2]]
                self.wrec_list[mesh[2]].conneced_wrec_dict[mesh[0]] = self.wrec_list[mesh[0]]
                self.wrec_list[mesh[2]].conneced_wrec_dict[mesh[1]] = self.wrec_list[mesh[1]]

                self.wrec_list[mesh[0]].belonged_mesh.append(mesh)
                self.wrec_list[mesh[1]].belonged_mesh.append(mesh)
                self.wrec_list[mesh[2]].belonged_mesh.append(mesh)

        # 力を計算する
        i = 0
        while i < HOW_MANY_MOVES:
            for wrec in self.wrec_list:
                wrec.calculate_fs()
                wrec.calculate_fr()
                wrec.calculate_all_f()
            self.move() #ここで、移動後次回のfaは計算ずみ。
            i += 1

        # word_positions_in_pic は wrec_set.wrec_list のINDEX順番と同じ。
        for i, wrec in zip( range(len(self.wrec_list)), self.wrec_list):
            if i == 0:
                word_positions_in_pic = np.array([[wrec.p_c[0], wrec.p_c[1]]])
            if i > 0:
                a_2d_ex = np.array([[wrec.p_c[0], wrec.p_c[1]]])
                word_positions_in_pic = np.append(word_positions_in_pic, a_2d_ex, axis=0)

        print(word_positions_in_pic)
        self.x_max = np.max(word_positions_in_pic, axis = 0)[0]
        self.y_max = np.max(word_positions_in_pic, axis = 0)[1]
        self.x_min = np.min(word_positions_in_pic, axis = 0)[0]
        self.y_min = np.min(word_positions_in_pic, axis = 0)[1]

        self.word_positions_in_pic = word_positions_in_pic

        # 描画様にデータをcsvで排出
        i = 0
        for wrec in self.wrec_list:
            if i == 0:
                tmp_df = pd.DataFrame(\
                          [[wrec.word,\
                            wrec.p_c[0],\
                            wrec.p_c[1],\
                            wrec.p_tl[0],\
                            wrec.p_tl[1],\
                            wrec.p_tr[0],\
                            wrec.p_tr[1],\
                            wrec.p_bl[0],\
                            wrec.p_bl[1],\
                            wrec.p_br[0],\
                            wrec.p_br[1],\
                            wrec.size,\
                            wrec.color\
                         ]]\
                        )

                tmp_df.columns = ['word', 'p_c_x', 'p_c_y', 'p_tl_x', 'p_tl_y', 'p_tr_x', 'p_tr_y', 'p_bl_x', 'p_bl_y', 'p_br_x', 'p_br_y', 'size', 'color']
            
            if i > 0:
                tmp_df = tmp_df.append({'word'    : wrec.word,\
                                        'p_c_x'   : wrec.p_c[0],\
                                        'p_c_y'   : wrec.p_c[1],\
                                        'p_tl_x'  : wrec.p_tl[0],\
                                        'p_tl_y'  : wrec.p_tl[1],\
                                        'p_tr_x'  : wrec.p_tr[0],\
                                        'p_tr_y'  : wrec.p_tr[1],\
                                        'p_bl_x'  : wrec.p_bl[0],\
                                        'p_bl_y'  : wrec.p_bl[1],\
                                        'p_br_x'  : wrec.p_br[0],\
                                        'p_br_y'  : wrec.p_br[1],\
                                        'size'    : wrec.size,\
                                        'color'   : wrec.color\
                                        } , ignore_index=True)

            i = i+1
        SAVE_PATH = './Bokeh/CSVs/afrer_forced_output_' + str(DRAW_INDEX) + '.csv'
        tmp_df.to_csv(SAVE_PATH, index=False)

    def draw_word_crowd(self, DRAW_INDEX):
        #ワードクラウド描画
        X_SIZE = int(self.x_max - self.x_min + 400)
        Y_SIZE = int(self.y_max - self.y_min + 400)

        adjust_x = self.x_min - 200
        adjust_y = self.y_min - 200

        campus = Image.new('RGB', (X_SIZE, Y_SIZE), (128, 128, 128))
        draw = ImageDraw.Draw(campus)

        for wrec in self.wrec_list:    
            if wrec.color == "RED":
                draw.rectangle((wrec.p_tl[0] - adjust_x, \
                                wrec.p_tl[1] - adjust_y, \
                                wrec.p_br[0] - adjust_x, \
                                wrec.p_br[1] - adjust_y), \
                                fill=(240, 0, 0), outline=(255, 255, 255))
                print(wrec.p_tl[0])

            if wrec.color == "BLUE":
                draw.rectangle((wrec.p_tl[0] - adjust_x, \
                                wrec.p_tl[1] - adjust_y, \
                                wrec.p_br[0] - adjust_x, \
                                wrec.p_br[1] - adjust_y), \
                                fill=(0, 0, 240), outline=(255, 255, 255))

            if wrec.color == "PURPLE":
                draw.rectangle((wrec.p_tl[0] - adjust_x, \
                                wrec.p_tl[1] - adjust_y, \
                                wrec.p_br[0] - adjust_x, \
                                wrec.p_br[1] - adjust_y), \
                                fill=(150, 0, 150), outline=(255, 255, 255))
                
            if wrec.color == "NO":
                draw.rectangle((wrec.p_tl[0] - adjust_x, \
                                wrec.p_tl[1] - adjust_y, \
                                wrec.p_br[0] - adjust_x, \
                                wrec.p_br[1] - adjust_y), \
                                fill=(50, 50, 50), outline=(255, 255, 255))

            if wrec.color == "RED" or wrec.color == "BLUE" or wrec.color == "PURPLE" or wrec.color == "NO":
                ttfontname = "./logotypejp_mp_m_1.1.ttf"
                fontsize = wrec.size
                font = ImageFont.truetype(ttfontname, fontsize)
                text_position_x = wrec.p_tl[0] - adjust_x
                text_position_y = wrec.p_tl[1] - adjust_y
                textRGB = (20, 20, 20)
                text = wrec.word
                draw.text((text_position_x, text_position_y), text, fill=textRGB, font=font)

            if wrec.color == "Thumbnail":
                url = 'http://i.ytimg.com/vi/' + wrec.word + "/mqdefault.jpg"
                response = requests.get(url)
                image = response.content
                file_name = "Thumbnail/" + wrec.word  + ".jpeg"
                with open(file_name, "wb") as aaa:
                    aaa.write(image)
                img = Image.open("Thumbnail/" +  wrec.word + ".jpeg")
                img_resize = img.resize( (int((wrec.p_br[0] - wrec.p_bl[0]) / 2), int((wrec.p_bl[1] - wrec.p_tl[1]) / 2)) )
                campus.paste(img_resize, (int(wrec.p_c[0]), int(wrec.p_c[1])) )
        
        campus.save('./Images/pillow_imagedraw.jpg', quality=95)
        tri_2 = Delaunay(self.word_positions_in_pic)
        fig_2 = delaunay_plot_2d(tri_2)
        fig_2.savefig('./Images/scipy_matplotlib_delaunay_after.png')

'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y)
ax.set_title('first scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig('./Images/scatter.png')
'''