import math

import numpy as np
from xgboost import XGBClassifier


def construct_xgb(param):
    bound_list = [[1, 25],
                  [0.001, 0.1],
                  [50, 1200],
                  [0.05, 0.9],
                  [1, 9],
                  [0.5, 1],
                  [0.5, 1],
                  [0.5, 1],
                  [0.1, 0.9],
                  [0.01, 0.1]]
    para_list = param.detach().numpy()
    # print(para_list[0])

    para_ = []
    for i in range(10):
        para = para_list[i]
        if i == 0 or i == 2 or i == 4:
            para = int(para)
        if para > bound_list[i][1]:
            para = bound_list[i][1]
        elif para < bound_list[i][0]:
            para = bound_list[i][0]
        para_.append(para)
        # print('para_', para_)

    model = XGBClassifier(max_depth=para_[0],
                          learning_rate=para_[1],
                          n_estimators=para_[2],
                          gamma=para_[3],
                          min_child_weight=para_[4],
                          subsample=para_[5],
                          colsample_bytree=para_[6],
                          colsample_bylevel=para_[7],
                          reg_alpha=para_[8],
                          reg_lambda=para_[9],
                          use_label_encoder=False)
    return model