from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

from PIL import Image, ImageDraw, ImageFont
import requests
import os
import MeCab


KR = 100000

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

        self.conneced_wrec_dict = {}
        self.fs = np.zeros(2)
        self.fr = np.zeros(2)
        self.fa = np.zeros(2)
        self.f_all = np.zeros(2)
    
    def calculate_fs(self):
        self.fs = np.zeros(2)
        for wrec in self.conneced_wrec_dict.values():
            d = np.linalg.norm(self.p_c - wrec.p_c)
            one_fs = self.size * wrec.size * d\
                               * ((wrec.p_c - self.p_c)  / np.linalg.norm(wrec.p_c - self.p_c))
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
        s = 1

    def calculate_all_f(self):
        self.f_all = self.fs + self.fr + self.fa



class Wrecset:
    def __init__(self, path):
        self.wrec_list = []
        self.df = pd.read_csv('./CSVs/positions_corners_size_csv_out.csv')
        row_no = len(self.df)
        i = 0
        while i < row_no:
            wrec = Wrec(self.df.iat[i, 0], self.df.iat[i, 1], self.df.iat[i, 2], self.df.iat[i, 3], \
                        self.df.iat[i, 4], self.df.iat[i, 5], self.df.iat[i, 6], self.df.iat[i, 7],\
                        self.df.iat[i, 8], self.df.iat[i, 9], self.df.iat[i, 10], self.df.iat[i, 11], self.df.iat[i, 12],\
                        i)  
            self.wrec_list.append(wrec)
            i += 1  

    def move(self):
        for wrec in self.wrec_list:
            wrec.p_c  = wrec.p_c  + wrec.f_all / 10000 #　とりあえず10,000
            wrec.p_tl = wrec.p_tl + wrec.f_all / 10000 #　とりあえず10,000
            wrec.p_tr = wrec.p_tr + wrec.f_all / 10000 #　とりあえず10,000
            wrec.p_bl = wrec.p_bl + wrec.f_all / 10000 #　とりあえず10,000
            wrec.p_br = wrec.p_br + wrec.f_all / 10000 #　とりあえず10,000
            # print(wrec.f_all / 10000)
#ドロネー三角分割
wrec_set = Wrecset('./CSVs/positions_corners_size_csv_out.csv')

#　同じ値のものはちょっとずらず
for wrec_1 in wrec_set.wrec_list:
    for wrec_2 in wrec_set.wrec_list:
        if wrec_1 == wrec_2:
            pass
        if np.all(wrec_1.p_c == wrec_2.p_c):
            wrec_1.p_c = wrec_1.p_c + random.random() * 5 #とりあえず5なだけ

# word_positions_in_pic は wrec_set.wrec_list のINDEX順番と同じ。
for i, wrec in zip( range(len(wrec_set.wrec_list)), wrec_set.wrec_list):
    if i == 0:
        word_positions_in_pic = np.array([[wrec.p_c[0], wrec.p_c[1]]])
    if i > 0:
        a_2d_ex = np.array([[wrec.p_c[0], wrec.p_c[1]]])
        word_positions_in_pic = np.append(word_positions_in_pic, a_2d_ex, axis=0)

print(word_positions_in_pic)
tri = Delaunay(word_positions_in_pic)
fig = delaunay_plot_2d(tri)
fig.savefig('./Images/scipy_matplotlib_delaunay_before.png')
#print(tri.simplices)

for mesh in tri.simplices:
    if mesh[0] == wrec_set.wrec_list[mesh[0]].index_num:#  一応一致しているかを確認
        wrec_set.wrec_list[mesh[0]].conneced_wrec_dict[mesh[1]] = wrec_set.wrec_list[mesh[1]]
        wrec_set.wrec_list[mesh[0]].conneced_wrec_dict[mesh[2]] = wrec_set.wrec_list[mesh[2]]

        wrec_set.wrec_list[mesh[1]].conneced_wrec_dict[mesh[0]] = wrec_set.wrec_list[mesh[0]]
        wrec_set.wrec_list[mesh[1]].conneced_wrec_dict[mesh[2]] = wrec_set.wrec_list[mesh[2]]

        wrec_set.wrec_list[mesh[2]].conneced_wrec_dict[mesh[0]] = wrec_set.wrec_list[mesh[0]]
        wrec_set.wrec_list[mesh[2]].conneced_wrec_dict[mesh[1]] = wrec_set.wrec_list[mesh[1]]

# 力を計算する
for wrec in wrec_set.wrec_list:
    wrec.calculate_fs()
    wrec.calculate_fr()

i = 0
while i < 2000:
    wrec_set.move()
    for wrec in wrec_set.wrec_list:
        wrec.calculate_fs()
        wrec.calculate_fr()
        wrec.calculate_all_f()
    i += 1

# word_positions_in_pic は wrec_set.wrec_list のINDEX順番と同じ。
for i, wrec in zip( range(len(wrec_set.wrec_list)), wrec_set.wrec_list):
    if i == 0:
        word_positions_in_pic = np.array([[wrec.p_c[0], wrec.p_c[1]]])
    if i > 0:
        a_2d_ex = np.array([[wrec.p_c[0], wrec.p_c[1]]])
        word_positions_in_pic = np.append(word_positions_in_pic, a_2d_ex, axis=0)

print(word_positions_in_pic)

#ワードクラウド描画
X_SIZE=3000
Y_SIZE=3000

campus = Image.new('RGB', (X_SIZE, Y_SIZE), (128, 128, 128))
draw = ImageDraw.Draw(campus)

for wrec in wrec_set.wrec_list:    
    if wrec.color == "RED":
        draw.rectangle((wrec.p_tl[0], \
                        wrec.p_tl[1], \
                        wrec.p_br[0], \
                        wrec.p_br[1]), \
                        fill=(240, 0, 0), outline=(255, 255, 255))
        print(wrec.p_tl[0])

    if wrec.color == "BLUE":
        draw.rectangle((wrec.p_tl[0], \
                        wrec.p_tl[1], \
                        wrec.p_br[0], \
                        wrec.p_br[1]), \
                        fill=(0, 0, 240), outline=(255, 255, 255))

    if wrec.color == "PURPLE":
        draw.rectangle((wrec.p_tl[0], \
                        wrec.p_tl[1], \
                        wrec.p_br[0], \
                        wrec.p_br[1]), \
                        fill=(150, 0, 150), outline=(255, 255, 255))
        
    if wrec.color == "NO":
        draw.rectangle((wrec.p_tl[0], \
                        wrec.p_tl[1], \
                        wrec.p_br[0], \
                        wrec.p_br[1]), \
                        fill=(50, 50, 50), outline=(255, 255, 255))
    
    ttfontname = "./logotypejp_mp_m_1.1.ttf"
    fontsize = wrec.size * 6
    font = ImageFont.truetype(ttfontname, fontsize)
    text_position_x = wrec.p_tl[0]
    text_position_y = wrec.p_tl[1]
    textRGB = (20, 20, 20)
    text = wrec.word
    draw.text((text_position_x, text_position_y), text, fill=textRGB, font=font)

    campus.save('./Images/pillow_imagedraw.jpg', quality=95)

tri_2 = Delaunay(word_positions_in_pic)
fig_2 = delaunay_plot_2d(tri_2)
fig_2.savefig('./Images/scipy_matplotlib_delaunay_after.png')
'''
for wrec in wrec_set.wrec_list:
    print(wrec.conneced_wrec_dict)
'''


'''
x = word_positions_in_pic[:, 0]
y = word_positions_in_pic[:, 1]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y)
ax.set_title('first scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig('./Images/scatter.png')
'''