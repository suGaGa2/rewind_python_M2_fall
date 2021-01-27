from myModule import *
from myModuleForceModel import *

#!!!!!!!!!!!  ここでインターバル数を決める。最低でも3以上にすること　！！     ！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!
TSET_INTERVAL_NUM = 45
# 出現回数が閾値以降のものを残す時に、何個wordを残す(抽出する)かを決める値。 !!!!!!!!!!!!!!!!!!!!!!!!!
WORDS_NUM_IN_A_CLOUD = 36

# ワードクラウドとして描画する t_elementのINDEX
# DRAW_INDEX = 14 ←　for文で回すようになったから，ここは不要

DRAW＿INDEX = 0
while DRAW_INDEX  <  TSET_INTERVAL_NUM:
    # watchs
    watchs = Watchs('./CSVs/output.csv')

    # watch_listを作成
    watchs.construct_watch_list_all()

    # t_setを作成・初期化
    t_set = Tset(watchs.watch_start_datetime_all, watchs.watch_end_datetime_all)
    t_set.set_interval_num(TSET_INTERVAL_NUM) 
    # t_setを、t_elementに、start_datetime, end_datetime, index(0オリジン)を登録して生成。
    t_set.construct_element_list()

    # ワードリストを作る。
    watchs.tag_each_watch()

    # t_setの各t_elementに対するword_count_dictを作成
    t_set.construct_t_element_word_count_dict_2nd(watchs, WORDS_NUM_IN_A_CLOUD)

    # w_set作る。初期情報登録
    w_set = Wset()
    w_set.set_elements_dict(t_set)
    w_set.set_element_dict_importance_vec(t_set)

    # Initial positionをMDSで計算
    w_set.set_words_initital_position(t_set)

    # t_setの各t_elementの抽出されたwordに諸々の情報を登録
    t_set.set_t_element_extracted_w_info_dic(watchs, w_set)

    # サムネイルを描画するために、指定時間範囲の視聴頻度、タグのポジションからのサムネイルの配置を考える。
    t_set.set_t_element_extracted_watch_list(watchs)

    #print(t_set.elements_list[5].word_count_dict)

    # ①ワードクラウドの表示 
    # → positions_corners_size_csv_out.csv　に書き出し
    t_set.draw_word_crowd(DRAW_INDEX, X_SIZE=600, Y_SIZE=600)

    # ②サムネイル クラウドの表示 
    # →　positions_corners_size_csv_out.csv に 
    # →　サムネイルの位置を追加して withThumb_positions_corners_size_csv_out.csv に追加
    t_set.draw_thumbnail_crowd_with_word(DRAW_INDEX, X_SIZE=600, Y_SIZE=600)


    # 力学モデル！！！！
    #wrec_set = Wrecset("./CSVs/positions_corners_size_csv_out.csv")
    wrec_set = Wrecset("./CSVs/withThumb_positions_corners_size_csv_out" + "_"+ str(DRAW＿INDEX)+".csv")
    wrec_set.force_model(DRAW_INDEX)

    DRAW＿INDEX += 1

# significance curveの表示
t_set.draw_significance_curve(TSET_INTERVAL_NUM)

'''
app = QApplication(sys.argv)
main_window = MainWindow("pillow_imagedraw.jpg", "S_X.png", \
                         t_set.elements_list[DRAW_INDEX].start_datetime,\
                         t_set.elements_list[DRAW_INDEX].end_datetime)
main_window.show()
sys.exit(app.exec_())
'''




