from hm.ml.houserent.data_manager import generate_feature_csv
from hm.ml.houserent.rent_main import model_train


def main():
    # 数据处理，生成特征数据文件
    generate_feature_csv()
    # 模型训练
    model_train()


if __name__ == '__main__':
    main()
