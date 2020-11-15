#MeCabをインストール
import MeCab
import re

import requests

from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np


w = h = 360
n = 6
np.random.seed(0)
pts = np.random.randint(0, w, (n, 2))

print(pts)
# [[172  47]
#  [117 192]
#  [323 251]
#  [195 359]
#  [  9 211]
#  [277 242]]

print(type(pts))
# <class 'numpy.ndarray'>

print(pts.shape)
# (6, 2)

tri = Delaunay(pts)

print(type(tri))
# <class 'scipy.spatial.qhull.Delaunay'>

fig = delaunay_plot_2d(tri)
fig.savefig('scipy_matplotlib_delaunay.png')






'''
url = 'http://i.ytimg.com/vi/' + "TzaYNiT_CRg" + "/mqdefault.jpg"
response = requests.get(url)

image = response.content

file_name = "TzaYNiT_CRg.jpeg"

with open(file_name, "wb") as aaa:
    aaa.write(image)
'''

'''
#形態素解析したい文章
data = "すもももももももものうち"

path = "./mecab_test_imput.csv"

with open(path) as f:
    for s_line in f:
        mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd').parse(s_line)
        lines = mecab.split('\n')

        nounAndVerb = []
        for line in lines:
            feature = line.split('\t')
            if len(feature) == 2:#'EOS'と''を省く
                info = feature[1].split(',')
                hinshi = info[0]
                if hinshi in ('名詞', '動詞'):
                    nounAndVerb.append(info[6])
                    print(nounAndVerb)
'''
