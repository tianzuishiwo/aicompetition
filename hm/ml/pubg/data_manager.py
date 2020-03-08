import pandas as pd
import numpy as np
import time

from hm.ml.pubg.pubg_config import *
from common.my_decorator import *

COL_ID = 'Id'  # 用户id ---> 统计多少玩家
COL_groupId = 'groupId'  # 所处小队id
COL_matchId = 'matchId'  # 该场比赛id
COL_assists = 'assists'  # 助攻数
COL_boosts = 'boosts'  # 使用能量,道具数量
COL_damageDealt = 'damageDealt'
COL_DBNOs = 'DBNOs'  # 击倒敌人数量
COL_headshotKills = 'headshotKills'  # 爆头击杀 ---> 爆头率
COL_heals = 'heals'  # 使用治疗药品数量
COL_killPlace = 'killPlace'  # 本场比赛杀敌排行
COL_killPoints = 'killPoints'  # Elo杀敌排名
COL_kills = 'kills'
COL_killStreaks = 'killStreaks'
COL_longestKill = 'longestKill'  # 最远杀敌距离
COL_matchDuration = 'matchDuration'
COL_matchType = 'matchType'  # 比赛类型(小组人数) “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”
COL_maxPlace = 'maxPlace'  # 本局最差名次
COL_numGroups = 'numGroups'  # 小组数量
COL_rankPoints = 'rankPoints'  # Elo排名
COL_revives = 'revives'  # 救活队员的次数
COL_rideDistance = 'rideDistance'
COL_roadKills = 'roadKills'
COL_swimDistance = 'swimDistance'
COL_teamKills = 'teamKills'  # 杀死队友的次数
COL_vehicleDestroys = 'vehicleDestroys'
COL_walkDistance = 'walkDistance'
COL_weaponsAcquired = 'weaponsAcquired'
COL_winPoints = 'winPoints'  # 胜率Elo排名
COL_winPlacePerc = 'winPlacePerc'  # 百分比排名
# COL_ = ''
COL_playersJoined = 'playersJoined'
COL_totalDistance = 'totalDistance'
COL_killwithoutMoving = 'killwithoutMoving'
COL_groupId_cat = 'groupId_cat'
COL_matchId_cat = 'matchId_cat'
# COL_ = ''
# COL_ = ''

"""
字符串转唯一标识码：id groupId matchId
剔除异常值：
assists>35,boosts>50,damageDealt>1000,DBNOs>35



"""


class BaseDataHandle(object):
    def __init__(self, df):
        self.data = df
        self.handle()

    def handle(self):
        pass

    def get_data(self):
        return self.data

    def pshape(self):
        print(self.data.shape)


class MissValuer(BaseDataHandle):
    """缺失值处理"""

    # train = train.drop(2744604) 'winPlacePerc'的2744604有数据缺失

    def handle(self):
        pass


class AbnormalValuer(BaseDataHandle):
    """剔除异常值"""

    @caltime_p1('剔除异常值')
    def handle(self):
        self.data.drop(self.data[self.data['kills'] > 30].index, inplace=True)
        self.drop_kill_without_moving()

    @caltime_p1('增加列：总共移动距离 （并去掉不移动还能杀人的）')
    def drop_kill_without_moving(self):
        self.data[COL_totalDistance] = self.data[COL_rideDistance] + self.data[COL_walkDistance] + self.data[
            COL_swimDistance]
        self.data[COL_killwithoutMoving] = (self.data[COL_kills] > 0) & (self.data[COL_totalDistance] == 0)
        self.data.drop(self.data[self.data[COL_killwithoutMoving] == True].index, inplace=True)


class FeatureExtractor(BaseDataHandle):
    """特征提取"""

    @caltime_p1('特征提取')
    def handle(self):
        self.add_playersJoined()
        self.match_type_onehot()
        self.str_castto_int()

    @caltime_p1('增加列：每场参赛人数（并去掉<75人）')
    def add_playersJoined(self):
        self.data[COL_playersJoined] = self.data.groupby(COL_matchId)[COL_matchId].transform("count")
        self.data[COL_playersJoined].sort_values()
        # self.data.drop(self.data[self.data[COL_playersJoined] < 15].index, inplace=True)
        print('暂未删除playersJoined<75人')

    @caltime_p1('比赛类型onehot编码')
    def match_type_onehot(self):
        self.data = pd.get_dummies(self.data, columns=[COL_matchType])

    @caltime_p1('groupId,matchId列类型转换')
    def str_castto_int(self):
        self._real_str_castto_int(COL_groupId, COL_groupId_cat)
        self._real_str_castto_int(COL_matchId, COL_matchId_cat)

    # @caltime_p3('特征提取')
    def _real_str_castto_int(self, col_source, col_target):
        self.data[col_source] = self.data[col_source].astype('category')
        self.data[col_target] = self.data[col_source].cat.codes
        self.data.drop([col_source], axis=1, inplace=True)


class DataManager(object):

    def __init__(self):
        self.data = None
        self.load_data()
        self.raw_shape = self.data.shape

    @caltime_p1('加载数据集')
    def load_data(self):
        self.data = pd.read_csv(USE_TRAIN_PATH)

    @caltime_p1('生成特征数据')
    def generate_feature_df(self):
        self.print_info(self.data)
        self.data = MissValuer(self.data).get_data()
        self.data = AbnormalValuer(self.data).get_data()
        self.data = FeatureExtractor(self.data).get_data()

    def auto(self):
        self.generate_feature_df()
        # self.data = self.sample() # 数据截取
        self.print_info(self.data)
        self.save_feature()

    @caltime_p1('保存特征数据')
    def save_feature(self):
        self.data.to_csv(self.get_new_csv_path(self.raw_shape, self.data.shape))

    def print_info(self, df):
        print(df.shape)
        print(df.columns)
        # print(df.info())

    def pshape(self):
        print(self.data.shape)

    def get_new_csv_path(self, raw_shape, new_shape):
        # time-1000_23_to_333_45.csv
        csv_new = f'{str(int(time.time()))}-{raw_shape[0]}_{raw_shape[1]}_to_{new_shape[0]}_{new_shape[1]}.csv'
        print(f'生成特征数据文件： {csv_new}')
        path = PUBG_TRAIN_PATH + csv_new
        return path

    @caltime_p1('数据截取')
    def sample(self):
        count = 10000
        return self.data.sample(count)


@caltime_p0('数据处理全部流程')
def test_data():
    data = DataManager()
    data.auto()


def main():
    test_data()


if __name__ == '__main__':
    main()
