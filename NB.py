# 测试用
from Data import DataUtil
import numpy as np
class NaiveBayse:
    if __name__ == '__main__':
        from NB import NaiveBayse
        NB = NaiveBayse()
        NB.dealdata("C:\\balloon1.0.txt")
    def __init__(self):
        self._x = self._y = None # 数据集合
        self._dics_x = self._dics_y = None # 数据字典
        self._possibilities = None # 先验概率
        self._func = None # 决策函数
        self._count_y = None # y取值和出现次数
        self._features_count = None # 各特征的数量
        self._condition_p = None # 条件概率


    def dealdata(self, path):
        dx, dy = DataUtil.getdata(path)
        self._dics_y = {_y : i for i, _y in enumerate(set(dy))}
        self._dics_x = [{_x : i for i, _x in enumerate(set(x))} for x in dx.T]
        self._y = [self._dics_y[y] for y in dy]
        self._x = [[self._dics_x[x_idx][x] for x_idx, x in enumerate(sample)] for sample in dx]
        self._count_y = np.bincount(self._y)
        self._possibilities = [(yy + 1) / (sum(self._count_y) + len(self._count_y)) for yy in self._count_y]
        self._features_count = [len(set(feature)) for feature in np.array(self._x).T]
        y = np.array(self._y)
        x = np.array(self._x)
        feats = [y == value for value in range(len(self._count_y))]
        label_x = [x[ci].T for ci in feats]
        for ci in feats:
            print(ci)
            print("  ")
            print(x[ci])

        print()