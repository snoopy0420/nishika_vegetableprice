import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
import warnings
from util import Logger

import itertools
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # RAWデータ格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME'] # モデルの格納場所
REMOVE_COLS = yml['SETTING']['REMOVE_COLS']


#### preprocessing関数を定義 ##########################################################

# bininng
def get_bins(all_df):
    # bin_edges = [-1, 25, 45, np.inf]
    # all_df["bin_age"] = pd.cut(all_df["age"], bins=bin_edges, labels=["young", "middle", "senior"]).astype("object")
    bin_edges = [-1, 20, 30, 40, 50, 60, np.inf]
    all_df["bin_general"] = pd.cut(all_df["age"], bins=bin_edges, labels=["10's", "20's", "30's", "40's", "50's", "60's"]).astype("object")
    return all_df

def get_cross_cate_features(all_df):
    obj_cols = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship', 
        'race', 
        'sex',
        # 'native-country',
        # 'bin_age',
        'bin_general'
    ]
    for cols in itertools.combinations(obj_cols, 2):
        all_df["{}_{}".format(cols[0], cols[1])] = all_df[cols[0]] + "_" + all_df[cols[1]]
    return all_df

def get_cross_num_features(all_df):
    all_df["prod_age_educationnum"] = all_df["age"] * all_df["education-num"]
    all_df["ratio_age_educationnum"] = all_df["age"] / all_df["education-num"]
    return all_df

def get_agg_features(all_df):
    cate_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        # "native-country"
    ]
    group_values = [
        "age",
        "education-num",
    ]
    for col in cate_cols:
        for group in group_values:
            all_df.loc[:, "mean_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).mean()[group]) # 平均
            all_df.loc[:, "std_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).std()[group])   # 分散
            all_df.loc[:, "max_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).max()[group])   # 最大値
            # all_df.loc[:, "min_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).min()[group])   # 最小値
            # all_df.loc[:, "nunique_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).nunique()[group])   # uniaue
            # all_df.loc[:, "median_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).median()[group])   # 中央値
    return all_df

def get_relative_features(all_df):
    cate_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        # "native-country"
    ]
    group_values = [
        "age",
        "education-num",
    ]
    # カテゴリごとの平均との差
    for col in cate_cols:
        for group in group_values:
            df = all_df.copy()
            df.loc[:, "mean_{}_{}".format(col, group)] = df[col].map(all_df.groupby(col).mean()[group]) # 平均
            all_df.loc[:, "{}_diff_{}".format(col, group)] = df[group] - df["mean_{}_{}".format(col, group)]
    return all_df

def get_freq_features(all_df):
    cate_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        # "sex",
        # "native-country"
    ]
    for col in cate_cols:
        freq = all_df[col].value_counts()
        # カテゴリの出現回数で置換
        all_df.loc[:, "freq_{}".format(col)] = all_df[col].map(freq)
    return all_df

def get_labelencoding(all_df):
    cols = all_df.dtypes[(all_df.dtypes=="object") | (all_df.dtypes=="category")].index
    for col in cols:
        le = LabelEncoder()
        all_df.loc[:, col] = le.fit_transform(all_df[col])
    return all_df




##### main関数を定義 ###########################################################################
def main():
    
    # データの読み込み
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    
    # preprocessingの実行
    df = get_bins(df)
    df = get_cross_cate_features(df)
    df = get_cross_num_features(df)
    df = get_agg_features(df)
    df = get_relative_features(df)
    df = get_freq_features(df)
    df = get_labelencoding(df)
    
    # trainとtestに分割
    train = df.iloc[:len(train), :]
    test = df.iloc[len(train):, :]

    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    
    # pickleファイルとして保存
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')
#     logger.info(f'train shape: {train.shape}, test shape, {test.shape}')
    
    # 生成した特徴量のリスト
    features_list = list(df.drop(columns=REMOVE_COLS).columns)  # 学習に不要なカラムは除外
    
    # 特徴量リストの保存
    # features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':
    
#     global logger
#     logger = Logger(MODEL_DIR_NAME + "create_features" + "/")

    main()
