#MeCabをインストール
import MeCab
import re

import requests
url = 'http://i.ytimg.com/vi/' + "TzaYNiT_CRg" + "/mqdefault.jpg"
response = requests.get(url)

image = response.content

file_name = "TzaYNiT_CRg.jpeg"

with open(file_name, "wb") as aaa:
    aaa.write(image)
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
