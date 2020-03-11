# from ml.analysis.data_analysis import BaseAnalysis
import os
# from ml.main.config import *
from common.my_decorator import *

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


# class ResearchQuestionData(BaseAnalysis):
class ResearchQuestionData(object):
    """
    深度学习开始了，发现学习节奏太快，这个项目没时间搞了，但是又不能不搞，
    折中方案：
        直接搬取学文老师代码，暂不抽取封装，能跑通即可
    """
    def __init__(self, data):
        self.data = data
        self.analisis()
        pass

    @caltime_p1('问题数据研究')
    def analisis(self):
        # self.divider()
        # self.about_home_direct()
        # self.divider()
        self.about_272()  # 2.7.2  同一个小区属于不同的区
        self.about_273()  # 2.7.3  同一个小区地铁线路不同的问题
        self.about_274()  # 2.7.4  研究一下位置和地铁线路的关系
        self.about_275()  # 2.7.5  研究一下位置和地铁站点的关系
        self.about_276()  # 2.7.6  研究一下小区名，位置，地铁线路，站点的关系
        self.about_277()  # 2.7.7  研究一下是否有换乘站的存在
        self.about_278()  # 2.7.8  研究一下每个位置的地铁线路数和站点数
        self.about_279()  # 2.7.9  研究一下位置缺失的样本地铁站点是否也是缺失的
        self.about_2710()  # 2.7.10  位置和区的关系校验
        self.about_28()  # 2.8  看一下小区名过多的问题

        # self.divider()
        # self.divider()
        # self.divider()
        # self.divider()
        # self.divider()
        pass

    @caltime_p1('房间朝向列有多个值问题')
    def about_home_direct(self):
        # self.col_head('房屋朝向')
        # self.col_value_counts('房屋朝向')  # 查看房屋朝向列有哪些值
        # self.add_new_home_direct()
        pass

    def add_new_home_direct(self):
        for i in range(5):
            self.data['朝向_' + str(i)] = self.data['房屋朝向'].map(lambda x: self._split(x, i))
        self.head(20)
        names = ["朝向_{}".format(i) for i in range(5)]
        self.data[names].info()
        pass

    def _split(self, text, i):
        """
        实现对字符串进行分割,并取出结果中下标i对应的值
        """
        items = text.split(" ")
        if i < len(items):
            return items[i]
        else:
            return np.nan

    @caltime_p1('2.7.2  同一个小区属于不同的区')
    def about_272(self):
        # self.head()
        # self.columns()
        neighbors1 = self.data[['小区名', '区', '位置']]
        print(neighbors1.shape)
        print(neighbors1.head())
        # 去掉'小区名','位置'两个列重复值后  有5292个不重复值
        neighbors1 = self.data[['小区名', '位置']].drop_duplicates()
        print(neighbors1.shape)
        # 去掉'小区名','位置'两个列重复值 ,同时删除缺失值  得,有5291个不重复值
        neighbors1 = self.data[['小区名', '位置']].drop_duplicates().dropna()
        print(neighbors1.shape)
        # neighbors1按照小区名分组后保留分组条数大于1的小区名
        count = neighbors1.groupby('小区名')['位置'].count()
        ids = count[count > 1].index
        print(ids)
        # 在原数据中筛选出这些小区的信息
        neighbors_has_problem = self.data[['小区名', '位置']
        ][self.data['小区名'].isin(ids)].sort_values(by='小区名')
        print(neighbors_has_problem.shape)
        print(neighbors_has_problem.head())
        # 找到每个小区的位置众数
        # 这里要注意x.mode有可能返回多个众数，所以用一个np.max拿到最值最大的众数作为最终的结果
        position_mode_of_neighbors = neighbors_has_problem.groupby(
            '小区名').apply(lambda x: np.max(x['位置'].mode()))
        # 位置缺失值就用这个数据来进行填充，
        # 对于已有的一个小区位于不同的位置，考虑到可能是因为小区太大导致，并不能认为是逻辑错误，保持不变
        print(position_mode_of_neighbors.head())
        pass

    def about_273(self):  # 2.7.3  同一个小区地铁线路不同的问题
        # 去掉'小区名','地铁线路'两个列重复之后  有3207个不重复值
        lines = self.data[['小区名', '地铁线路']].drop_duplicates().dropna()
        print(lines.shape)
        # 而有地铁的小区名只有3138个不重复值  说明有69个小区有多个地铁线路

        print(self.data[self.data['地铁线路'].notnull()].drop_duplicates(['小区名']).shape)
        # lines按照小区名分组后保留分组条数大于1的小区名   最终有多条地铁的小区有68个
        # 这个地铁线路分位置可能有关系  因为同一个小区位于不同的位置，地铁线路也有可能不同

        count = lines.groupby('小区名')['地铁线路'].count()
        ids = count[count > 1].index
        print(ids.shape)
        pass

    def about_274(self):  # 2.7.4  研究一下位置和地铁线路的关系
        self.data[['位置', '地铁线路']].drop_duplicates().dropna().head()
        # 去掉'位置','地铁线路'两个列重复之后  有184个不重复值

        pos_lines = self.data[['位置', '地铁线路']].drop_duplicates().dropna()
        print(pos_lines.shape)
        # 我们在来看一下有地铁的位置中有多少个不同的   120个
        pos_lines['位置'].value_counts().head()
        # pos_lines按照位置分组后保留分组条数大于1的位置  最终有多条地铁的位置有49个

        count = pos_lines.groupby('位置')['地铁线路'].count()
        ids = count[count > 1].index
        print(ids.shape)
        pass

    def about_275(self):  # 2.7.5  研究一下位置和地铁站点的关系
        # 去掉'位置','地铁站点'两个列重复之后  有337个不重复值

        pos_stations = self.data[['位置', '地铁站点']].drop_duplicates().dropna()
        print(pos_stations.shape)
        print(pos_stations.head())

        # 我们在来看一下有地铁的位置中有多少个不同的   120个

        pos_stations['位置'].value_counts().head()

        # pos_stations按照位置分组后保留分组条数大于1的位置  最终有多个站点的位置有97个

        count = pos_stations.groupby('位置')['地铁站点'].count()
        ids = count[count > 1].index
        print(ids.shape)
        pass

    def about_276(self):  # 2.7.6  研究一下小区名，位置，地铁线路，站点的关系
        # 去掉"小区名，位置，地铁线路，站点"四列重复之后  有3356个不重复值

        neighbor_pos_stations = self.data[['小区名', '位置',
                                       '地铁线路', '地铁站点']].drop_duplicates().dropna()
        print(neighbor_pos_stations.shape)
        # 看一下是否存在下小区名，位置一样的情况下，地铁线路不一样的情况

        # 可以看出：3356-3209=147条小区名，位置，地铁线路同样的情况下，地铁站点不一样
        # 3356-3147=209条小区名，位置一样，地铁线路不一样
        # 这种情况可能是因为数据错误，也有可能是实际情况，后面对此我们不做处理

        print(neighbor_pos_stations[['小区名', '位置', '地铁线路']
              ].drop_duplicates().dropna().shape)
        print(neighbor_pos_stations[['小区名', '位置']].drop_duplicates().dropna().shape)
        pass

    def about_277(self):  # 2.7.7  研究一下是否有换乘站的存在
        self.data[['地铁线路', '地铁站点']].head()
        self.data[['地铁线路', '地铁站点']].drop_duplicates(
        ).dropna().groupby('地铁站点').count().head()

        # 结果说明没有换乘站点存在，因为每个站点仅仅属于一条地铁线路

        self.data[['地铁线路', '地铁站点']].drop_duplicates(
        ).dropna().groupby('地铁站点').count().max(0)


        pass

    def about_278(self):  # 2.7.8  研究一下每个位置的地铁线路数和站点数
        # 每个位置的线路数 这个可以作为新特征加入

        a = self.data[['位置', '地铁线路']].drop_duplicates().dropna().groupby('位置').count()
        a.head()

        # 每个位置的站点数   也可以作为新特征加入

        b = self.data[['位置', '地铁站点']].drop_duplicates().dropna().groupby('位置').count()
        b.head()

        # 两者的相关性

        al = pd.concat([a, b], axis=1)
        al.head()

        al.corr()


        pass

    def about_279(self):  # 2.7.9  研究一下位置缺失的样本地铁站点是否也是缺失的
        self.data[["位置", "地铁站点", "地铁线路"]].head()

        # 发现存在地铁线路为缺失而位置缺失的情况   说明后面在填充位置缺失值的时候可以用地铁站点来进行填充

        pos_lines = self.data[['位置', '地铁站点']].drop_duplicates()

        pos_lines.head()

        pos_lines['位置'].isnull().sum()

        # 每个站点的位置数   也可以作为新特征加入

        self.data[['位置', '地铁站点']].drop_duplicates().dropna().groupby('地铁站点').count().head()




        pass

    def about_2710(self):  # 2.7.10  位置和区的关系校验

        # 查看是否存在一个位置率属于不同的区

        self.data[['位置', '区']].head()

        self.data[['位置', '区']].drop_duplicates().dropna().groupby('位置').count().head()

        # 说明每个位置仅仅属于一个区，不存在同一个位置属于两个区的现象

        self.data[['位置', '区']].drop_duplicates().dropna().groupby('位置').count().max()

        pass

    def about_28(self):  # 2.8  看一下小区名过多的问题

        self.data['小区名'].head()

        neighbors = self.data['小区名'].value_counts()

        neighbors.head()

        # 观察条目数超过50的小区有多少

        (neighbors > 50).sum()

        # 观察条目数超过100的小区有多少

        print((neighbors > 100).sum())

        pass
