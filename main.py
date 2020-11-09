
from myModule import *

#!!!!!!!!!!!  ここでインターバル数を決める。最低でも3以上にすること　！！     ！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!
TSET_INTERVAL_NUM = 15
# 出現回数が閾値以降のものを残す時に、何個wordを残す(抽出する)かを決める値。 !!!!!!!!!!!!!!!!!!!!!!!!!
WORDS_NUM_IN_A_CLOUD = 20
# ワードクラウドとして描画する t_elementのINDEX
DRAW_INDEX = 5 

# watchsを設定
watchs = Watchs('output.csv')
watchs.construct_watch_list_all()

# t_setを作成・初期化
t_set = Tset(watchs.watch_start_datetime_all, watchs.watch_end_datetime_all)
t_set.set_interval_num(TSET_INTERVAL_NUM) 
# t_setを、t_elementに、start_datetime, end_datetime, index(0オリジン)を登録して生成。
t_set.construct_element_list()



# t_setの各t_elementに対するword_count_dictを作成
t_set.construct_t_element_word_count_dict(watchs, WORDS_NUM_IN_A_CLOUD)

# w_set作る。初期情報登録
w_set = Wset()
w_set.set_elements_dict(t_set)
w_set.set_element_dict_importance_vec(t_set)

# Initial positionをMDSで計算
w_set.set_words_initital_position(t_set)

#t_setの各t_elementの抽出されたwordに諸々の情報を登録
t_set.set_t_element_extracted_w_info_dic(watchs, w_set)

# サムネクラウドの表示
t_set.draw_crowd(DRAW_INDEX)
# Significance curveの表示
t_set.draw_significance_curve(TSET_INTERVAL_NUM)



'''
app = QApplication(sys.argv)
main_window = MainWindow("pillow_imagedraw.jpg", "S_X.png", \
                         _START_DATETIME,\
                         _END_DATETIME,\
                        )
main_window.show()
sys.exit(app.exec_())
'''





