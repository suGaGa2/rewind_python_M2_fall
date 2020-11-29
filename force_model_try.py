from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

class Wrec:
    def __init__(self, word, p_c_x, p_c_y, p_tl_x, p_tl_y, p_tr_x, p_tr_y, p_bl_x, p_bl_y, p_br_x, p_br_y, size, i):
        self.index_num = i
        self.word = word

        self.p_c  = np.array([float(p_c_x),   float(p_c_y )])
        self.p_tl = np.array([float(p_tl_x),  float(p_tl_y)])
        self.p_tr = np.array([float(p_tr_x),  float(p_tr_y)])
        self.p_bl = np.array([float(p_bl_x),  float(p_bl_y)])
        self.p_br = np.array([float(p_br_x),  float(p_br_y)])

        self.seiz = int(size)

        self.conneced_wrec_dict = {}
        self.fs = np.zeros(2)
        self.fs = np.zeros(2)
        self.fa = np.zeros(2)
    
     def calculate_power(self):
         


class Wrecset:
    def __init__(self, path):
        self.wrec_list = []
        self.df = pd.read_csv('./CSVs/positions_corners_size_csv_out.csv')
        row_no = len(self.df)
        i = 0
        while i < row_no:
            wrec = Wrec(self.df.iat[i, 0], self.df.iat[i, 1], self.df.iat[i, 2], self.df.iat[i, 3], \
                        self.df.iat[i, 4], self.df.iat[i, 5], self.df.iat[i, 6], self.df.iat[i, 7],\
                        self.df.iat[i, 8], self.df.iat[i, 9], self.df.iat[i, 10], self.df.iat[i, 11],\
                        i)  
            self.wrec_list.append(wrec)
            i += 1  

#ドロネー三角分割
wrec_set = Wrecset('./CSVs/positions_corners_size_csv_out.csv')

#　同じ値のものはちょっとずらず
for wrec_1 in wrec_set.wrec_list:
    for wrec_2 in wrec_set.wrec_list:
        if np.all(wrec_1.p_c == wrec_2.p_c):
            wrec_1.p_c = wrec_1.p_c + random.random() * 5 #とりあえず5なだけ

# word_positions_in_pic は wrec_set.wrec_list のINDEX順番と同じ。
for i, wrec in zip( range(len(wrec_set.wrec_list)), wrec_set.wrec_list):
    if i == 0:
        word_positions_in_pic = np.array([[wrec.p_c[0], wrec.p_c[1]]])
    if i > 0:
        a_2d_ex = np.array([[wrec.p_c[0], wrec.p_c[1]]])
        word_positions_in_pic = np.append(word_positions_in_pic, a_2d_ex, axis=0)

tri = Delaunay(word_positions_in_pic)
fig = delaunay_plot_2d(tri)
fig.savefig('./Images/scipy_matplotlib_delaunay.png')
print(tri.simplices)

for mesh in tri.simplices:
    if mesh[0] == wrec_set.wrec_list[mesh[0]].index_num:#  一応一致しているかを確認
        wrec_set.wrec_list[mesh[0]].conneced_wrec_dict[mesh[1]] = wrec_set.wrec_list[mesh[1]]
        wrec_set.wrec_list[mesh[0]].conneced_wrec_dict[mesh[2]] = wrec_set.wrec_list[mesh[2]]

        wrec_set.wrec_list[mesh[1]].conneced_wrec_dict[mesh[0]] = wrec_set.wrec_list[mesh[0]]
        wrec_set.wrec_list[mesh[1]].conneced_wrec_dict[mesh[2]] = wrec_set.wrec_list[mesh[2]]

        wrec_set.wrec_list[mesh[2]].conneced_wrec_dict[mesh[0]] = wrec_set.wrec_list[mesh[0]]
        wrec_set.wrec_list[mesh[2]].conneced_wrec_dict[mesh[1]] = wrec_set.wrec_list[mesh[1]]

# 力を計算する

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