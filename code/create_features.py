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
import re

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

def change_to_num(all_df):
    """数値変数に変換
    """

    # 値の種類が1種類以下のものを除外　
    drop_cols = all_df.nunique()[all_df.nunique() <= 1].index
    all_df = all_df.drop(columns=drop_cols)

    # 最寄駅：距離（分）
    d = {
        "30分?60分": 45,
        "1H?1H30": 75,
        "2H?": 120,
        "1H30?2H": 105,
    }
    all_df["最寄駅：距離（分）"] = all_df["最寄駅：距離（分）"].replace(d)
    all_df["最寄駅：距離（分）"] = all_df["最寄駅：距離（分）"].astype(float)

    # 面積
    all_df.loc[:, '面積（㎡）'] = all_df["面積（㎡）"].replace({'2000㎡以上': "2000", "5000㎡以上": "5000"})
    all_df.loc[:, '面積（㎡）'] = all_df["面積（㎡）"].astype(int)
    all_df.loc[:, "面積_2000"] = all_df["面積（㎡）"] == 2000

    # 建築年
    d = {}
    for val in all_df["建築年"].value_counts().keys():
        if "平成" in val:
            _year = int(val.split("平成")[1].split("年")[0])
            year = 1988 + _year
        if "昭和" in val:
            _year = int(val.split("昭和")[1].split("年")[0])
            year = 1925 + _year
        if "令和" in val:
            _year = int(val.split("令和")[1].split("年")[0])
            year = 2018 + _year
        d[val] = year
    d["戦前"] = 1945
    all_df.loc[:, "建築年"] = all_df["建築年"].replace(d)

    # 取引時点
    all_df["取引年"] = all_df["取引時点"].apply(lambda x : int(x[:4]) if type(x)==str else np.nan)
    all_df["取引四半期"] = all_df["取引時点"].apply(lambda x : int(x[6]) if type(x)==str else np.nan)
    
    all_df.loc[:, "取引時点"] = all_df["取引時点"].apply(lambda x : re.sub('年第１四半期', '.00', x))
    all_df.loc[:, "取引時点"] = all_df["取引時点"].apply(lambda x : re.sub('年第２四半期', '.25', x))
    all_df.loc[:, "取引時点"] = all_df["取引時点"].apply(lambda x : re.sub('年第３四半期', '.50', x))
    all_df.loc[:, "取引時点"] = all_df["取引時点"].apply(lambda x : re.sub('年第４四半期', '.75', x))
    all_df.loc[:, "取引時点"] = all_df["取引時点"].apply(float)

    return all_df

def na_prep(all_df):
    """欠損値の処理
    """
    
    # 欠損値の数
    all_df.loc[:, "na_num"] = all_df.isnull().sum(axis=1).values
    
    na_cols = list(all_df.isnull().sum()[all_df.isnull().sum()>0].index)
    na_cols.remove("取引価格（総額）_log")
    # 欠損かどうかを表す二値変数
    for col in na_cols:
        all_df.loc[:, "{}_isna".format(col)] = all_df[col].isnull()
    
    # 欠損値の補完
    na_obj_col = all_df[na_cols].dtypes[all_df[na_cols].dtypes=="object"].index
    na_num_col = set(na_cols) - set(na_obj_col)
    for col in na_obj_col:
        all_df.loc[:,col] = all_df[col].fillna(all_df[col].mode()[0])
    for col in na_num_col:
        all_df.loc[:,col] = all_df[col].fillna(all_df[col].mean())
        
    return all_df

def madori_prep(all_df):
    """「間取り」の処理
    """
    all_df['L'] = all_df['間取り'].map(lambda x: 1 if 'Ｌ' in str(x) else 0)
    all_df['D'] = all_df['間取り'].map(lambda x: 1 if 'Ｄ' in str(x) else 0)
    all_df['K'] = all_df['間取り'].map(lambda x: 1 if 'Ｋ' in str(x) else 0)
    all_df['S'] = all_df['間取り'].map(lambda x: 1 if 'Ｓ' in str(x) else 0)
    all_df['R'] = all_df['間取り'].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
    all_df['OpenFloor'] = all_df['間取り'].map(lambda x: 1 if 'オープンフロア' in str(x) else 0)
    all_df['RoomNum'] = all_df['間取り'].map(lambda x: re.sub("\\D", "", str(x)))
    all_df.loc[:,'RoomNum'] = all_df['RoomNum'].map(lambda x:int(x) if x!='' else 0)
    all_df['TotalRoomNum'] = all_df[['L', 'D', 'K', 'S', 'R', 'RoomNum']].sum(axis=1)
    all_df['RoomNumRatio'] = all_df['RoomNum'] / all_df['TotalRoomNum']   
    
    return all_df

# ラベルエンコーディング
def get_labelencoding(all_df):
    cols = all_df.dtypes[all_df.dtypes=="object"].index
    for col in cols:
        all_df.loc[:, col] = all_df[col].fillna("NaN")
        le = LabelEncoder()
        all_df.loc[:, col] = le.fit_transform(all_df[col])

    return all_df



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




##### main関数を定義 ###########################################################################
def main():
    
    # データの読み込み
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    
    # preprocessingの実行
    df = change_to_num(df)
    df = na_prep(df)
    df = madori_prep(df)
    # df = get_labelencoding(df)
    
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
    print(features_list)
    
    # 特徴量リストの保存
    # features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt', encoding='utf-8') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':
    
#     global logger
#     logger = Logger(MODEL_DIR_NAME + "create_features" + "/")

    main()
