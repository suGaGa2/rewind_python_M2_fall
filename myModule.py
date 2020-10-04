import datetime
import numpy as np

class Watch:
    def __init__(self, video_title, video_id, channel_name, channel_id, watch_datetime):
        self.video_title = video_title
        self.video_id = video_id
        self.channel_name = channel_name
        self.channel_id = channel_id
        self.watch_datetime = datetime.datetime.strptime(str(watch_datetime), '%b %d, %Y, %I:%M:%S %p JST')


class Watchs:
    def __init__(self):
        self.watch_list_all = []
        self.watch_list_selected = []
        self.watch_start_datetime_all  = 0
        self.watch_end_datetime_all    = 0

    def set_watch_start_datetime_all(self):
        self.watch_start_datetime_all = self.watch_list_all[0].watch_datetime

    def set_watch_end_datetime_all(self):
        self.watch_end_datetime_all = self.watch_list_all[-1].watch_datetime

    def set_watch_list_selected(self, start_datetime, end_datetime):
        self.watch_list_selected.clear()
        for watch in self.watch_list_all:
            if watch.watch_datetime <= start_datetime and watch.watch_datetime >= end_datetime:
                self.watch_list_selected.append(watch)

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
        self.all_histgram_position_dict = {}
    
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
        self.channel_count_dict = {}    #{channelid : frequency,  channelid : frequency,} ソートされて、上位の選ばれたもの
        self.extracted_w_info_dict = {} # {channelid : w_info, id, w_info}
        self.index = i

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
    
    def add_elements(self, telement):
        self.elements_list.append(telement)
        

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
            for channel_id in t_element.channel_count_dict.keys():
                if not channel_id in self.elements_dict:
                    self.elements_dict[channel_id] = Welement(t_set.interval_num)
                
    
    def print_w_set(self):
        for w_element in self.elements_dict.items():
            print(w_element)
            print("****")
        

'''
-------------------------------------------------------------------------------
'''