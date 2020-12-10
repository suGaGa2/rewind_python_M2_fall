from myModuleForceModel import *

#wrec_set = Wrecset("./CSVs/positions_corners_size_csv_out.csv")
wrec_set = Wrecset("./CSVs/withThumb_positions_corners_size_csv_out.csv")
wrec_set.force_model()
wrec_set.draw_word_crowd()



