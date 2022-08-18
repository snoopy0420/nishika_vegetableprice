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
import datetime

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

# all_dfを受け取りall_dfを返す関数

def change_to_date(all_df):
    """datetimeに変換
    """
    if all_df["date"][0]==20041106:
        for i in ("max_temp_time", "min_temp_time"):
            all_df.loc[:, i] = pd.to_datetime(all_df[i], format="%Y/%m/%d %H:%M")
    else:
        all_df["date"] = pd.to_datetime(all_df["date"], format="%Y%m%d")

    return all_df


def merge_wea(all_df, wea):
    """卸売データと天候データをマージする
    """
    # 卸売データのエリア
    area_pairs = all_df["area"].unique()
    yasai_areas = set()
    for area_pair in area_pairs:
        areas = area_pair.split("_")
        yasai_areas = (yasai_areas | set(areas)) # 論理和
        
    # 天候データのエリア
    wea_areas = wea_df["area"].unique()

    # マッピングのための辞書を作成
    area_map = {}
    update_area_map = {
        '岩手':'盛岡','宮城':'仙台','静岡':'浜松','沖縄':'那覇','神奈川':'横浜','愛知':'名古屋','茨城':'水戸','北海道':'帯広','各地':'全国',
        '兵庫':'神戸','香川':'高松','埼玉':'熊谷','国内':'全国','山梨':'甲府','栃木':'宇都宮','群馬':'前橋','愛媛':'松山'
    }
    for yasai_area in yasai_areas:
        if (yasai_area in wea_areas):
            area_map[yasai_area] = yasai_area
        elif (yasai_area in update_area_map):
            area_map[yasai_area] = update_area_map[yasai_area]
        else:
            area_map[yasai_area] = "全国"

    # 卸売データのareaを置換
    all_df["area"] = all_df["area"].apply(lambda x: "_".join([area_map[i] for i in x.split("_")]))

    #＃ 天候データを処理
    # datetime型の平均の取り方がわからないので削除
    wea_df = wea_df.drop(columns=["max_temp_time","min_temp_time"])

    # wea_dfに全国を追加する
    agg_cols = [i for i in wea_df.columns if i not in ["area","date"]]
    tmp_df = wea_df.groupby(["date"])[agg_cols].agg(["mean"]).reset_index()

    new_cols = []
    for col1,col2 in tmp_df.columns:
        new_cols.append(col1)
    tmp_df.columns = new_cols

    tmp_df["area"] = "全国"
    tmp_df["date"] = wea_df[wea_df["area"]=="千葉"]["date"].values
    tmp_df = tmp_df[wea_df.columns]

    wea_df = pd.concat([wea_df, tmp_df])

    # 複数地点の平均を取る
    area_pairs = all_df["area"].unique()
    target_cols = [i for i in wea_df.columns if i not in("area","date")]
    date = wea_df[wea_df["area"]=="千葉"]["date"]
    area_pair_dfs = []
    for area_pair in area_pairs:
        areas = area_pair.split("_")
        # 全ての値が０のDFを作成
        base_tmp_df = pd.DataFrame(np.zeros(wea_df[wea_df["area"]=="千葉"][target_cols].shape), columns=target_cols)
        for area in areas:
            tmp_df = wea_df[wea_df["area"]==area].reset_index(drop=True)[target_cols]
            base_tmp_df = base_tmp_df.add(tmp_df)
        base_tmp_df /= len(areas)
        base_tmp_df["area"] = area_pair
        base_tmp_df["date"] = date.to_list()
        area_pair_dfs.append(base_tmp_df)

    area_pair_df = pd.concat(area_pair_dfs)
    all_df = pd.merge(all_df, area_pair_df, on=['date', 'area'], how='left')

    return all_df


def add_may(wea_df):
    """wea_dfに5月を追加する関数、ラグ特徴量生成に使用
    """
    # wea_dfに2022/05も追加
    start = datetime.datetime.strptime("2022-05-01", "%Y-%m-%d") # 5月の日付を取得
    may_date = pd.date_range(start, periods=31)
    for area in wea_df["area"].unique():
        # areaとdate意外NANの5月のdfを作る
        maywea_df = pd.DataFrame(columns=wea_df.columns,
                                data={"date":may_date,
                                      "area":area}
                                )
        # dtypesをfloat64に戻す
        cols = [i for i in maywea_df.columns if i not in ("date","area")]
        maywea_df[cols] = maywea_df[cols].astype("float64")
        # wea_dfとconcat
        wea_df = pd.concat([wea_df,maywea_df])
    # area,dateでソート
    wea_df = wea_df.sort_values(["area","date"])
    return wea_df

def get_lag_feat(all_df, wea_df, nshift):
    """単純ラグ特徴量
    """

    # mode_price, amount
    for value in ["mode_price", "amount"]:
        df_wide = all_df.pivot(index="date",columns="kind",values=value)
        df_wide_lag = df_wide.shift(nshift)
        df_long_lag = df_wide_lag.stack().reset_index()
        df_long_lag.columns = ["date", "kind", "{}_{}prev".format(value,nshift)]
        
        all_df = pd.merge(all_df, df_long_lag, on=['date', 'kind'], how='left')
        
    # wether
    # 5月を追加
    wea_df = add_may(wea_df)

    cols = [i for i in wea_df.columns if i not in ("area","date")]
    for value in cols:
        
        df_wide = wea_df.pivot(index="date",columns="area",values=value)
        df_wide_lag = df_wide.shift(nshift)
        df_long_lag = df_wide_lag.stack().reset_index()
        df_long_lag.columns = ["date", "area", "{}_{}prev".format(value,nshift)]
        
        all_df = pd.merge(all_df, df_long_lag, on=['date', 'area'], how='left')
        
    return all_df


def get_labelencoding(all_df):
    """ラベルエンコーディング
    """
    cols = all_df.dtypes[all_df.dtypes=="object"].index
    for col in cols:
        all_df.loc[:, col] = all_df[col].fillna("NaN")
        le = LabelEncoder()
        all_df.loc[:, col] = le.fit_transform(all_df[col])

    return all_df







def change_to_num(all_df):
    """datetimeに変換
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


def get_bins(all_df):
    """binning
    """
    # bin_edges = [-1, 25, 45, np.inf]
    # all_df["bin_age"] = pd.cut(all_df["age"], bins=bin_edges, labels=["young", "middle", "senior"]).astype("object")
    bin_edges = [-1, 20, 30, 40, 50, 60, np.inf]
    all_df["bin_general"] = pd.cut(all_df["age"], bins=bin_edges, labels=["10's", "20's", "30's", "40's", "50's", "60's"]).astype("object")
    return all_df

def get_cross_cate_features(all_df):
    """カテゴリ変数×カテゴリ変数
    """
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
    """数値変数×数値変数
    """
    all_df["prod_age_educationnum"] = all_df["age"] * all_df["education-num"]
    all_df["ratio_age_educationnum"] = all_df["age"] / all_df["education-num"]
    return all_df

def get_agg_features(all_df):
    """集約特徴量
    """
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
    """相対値
    """
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
    """frequency encoding
    """
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
    wea = pd.read_csv(RAW_DATA_DIR_NAME+"weather.csv")
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    
    # preprocessingの実行
    df = change_to_date(df)
    wea = change_to_date(wea)
    df = get_lag_feat(df)
    df = merge_wea(df,wea)
    df = get_lag_feat(df,wea,31)
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
