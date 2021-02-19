import sys
import os
import shutil
import datetime
import yaml
import json
import collections as cl
import warnings
from model_lgb import ModelLGB
from model_xgb import ModelXGB
from model_nn import ModelNN
from runner import Runner
from util import Submission, Util, threshold_optimization, optimized_f1
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize


# tensorflowの警告抑制
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')


# configの読み込み
CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']
TARGET = yml['SETTING']['TARGET']


def exist_check(path, run_name):
    """学習ファイルの存在チェックと実行確認
    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)

    # 通常の実行確認
    print('特徴量ディレクトリ:{} で実行しますか？[Y/n]'.format(FEATURE_DIR_NAME))
    x = input('>> ')
    if x != 'Y':
        print('終了します')
        sys.exit(0)

def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def my_makedirs_remove(path):
    """引数のpathディレクトリを新規作成する（存在している場合は削除→新規作成）
    path:ディレクトリ名
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_config(key_list, value_list, dir_name, run_name):
    """実装詳細を管理する
    どんな 特徴量/パラメータ/cv/setting/ で学習させたモデルかを管理するjsonファイルを出力する
    """
    def set_default(obj):
        """json出力の際にset型のオブジェクトをリストに変更する
        """
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    
    conf_dict = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        conf_dict[v] = data

    json_file = open(dir_name + run_name  + '_param.json', 'w')
    json.dump(conf_dict, json_file, indent=4, default=set_default)


# get label
# preds.pklに保存されているのは確率であるため、閾値の最適化後にラベルに変換
def get_label(train_y, train_preds, preds):
    bt = threshold_optimization(train_y, train_preds, optimized_f1)
    print(f"Best Threshold is {bt}")
    labels = preds >= bt
    return np.array(labels, dtype="int32")

    
# 以下で諸々の設定をする

def get_cv_info(random_state=42) -> dict:
    """CVの設定
    """
    # methodは[KFold, StratifiedKFold ,GroupKFold, StratifiedGroupKFold, CustomTimeSeriesSplitter, TrainTestSplit]から選択可能、
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFold or StratifiedGroupKFold の場合はcv_target_gr, cv_target_sfに対象カラム名を設定する
    cv_setting = {
        'method': 'KFold',
        'n_splits': 5,
        'random_state': random_state,
        'shuffle': True,
        'cv_target': None
    }
    return cv_setting


def get_run_name(cv_setting, model_type):
    """run名の作成、いじらなくてよい
    """
    run_name = model_type
    suffix = '_' + datetime.datetime.now().strftime("%m%d%H%M")
    model_info = ''
    run_name = run_name + '_' + cv_setting.get('method') + model_info + suffix
    return run_name


def get_file_info():
    """ファイル名の設定
    """
    file_setting = {
        'feature_dir_name': FEATURE_DIR_NAME,   # 特徴量の読み込み先ディレクトリ
        'model_dir_name': MODEL_DIR_NAME,       # モデルの保存先ディレクトリ
        'train_file_name': "train.pkl",          # 学習に使用するtrainファイル名
        'test_file_name': 'test.pkl',            # 予測に使用するtestファイル名
    }
    return file_setting


def get_run_info():
    """学習の設定
    """
    run_setting = {
        'target': TARGET,       # 目的変数
        'calc_shap': True,     # shap値を計算するか否か
        'save_train_pred': True,    # trainデータに対する予測値を保存するか否か,閾値の最適化に使用
        "hopt": False,           # パラメータチューニングするか否か
        "target_enc": True,     # target encoding をするか否か
        "cat_cols": [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship', 
            'bin_general',
            ]

    }
    return run_setting


if __name__ == '__main__':

    
##### 学習・推論 xgboost ###################################
 
    # 使用する特徴量の指定、features_list.txtからコピペ
    features = [
        'age',
        'workclass',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'bin_general',
        'workclass_education',
        'workclass_marital-status',
        'workclass_occupation',
        'workclass_relationship',
        'workclass_race',
        'workclass_sex',
        'workclass_bin_general',
        'education_marital-status',
        'education_occupation',
        'education_relationship',
        'education_race',
        'education_sex',
        'education_bin_general',
        'marital-status_occupation',
        'marital-status_relationship',
        'marital-status_race',
        'marital-status_sex',
        'marital-status_bin_general',
        'occupation_relationship',
        'occupation_race',
        'occupation_sex',
        'occupation_bin_general',
        'relationship_race',
        'relationship_sex',
        'relationship_bin_general',
        'race_sex',
        'race_bin_general',
        'sex_bin_general',
        'prod_age_educationnum',
        'ratio_age_educationnum',
        'mean_workclass_age',
        'std_workclass_age',
        'max_workclass_age',
        'mean_workclass_education-num',
        'std_workclass_education-num',
        'max_workclass_education-num',
        'mean_education_age',
        'std_education_age',
        'max_education_age',
        'mean_education_education-num',
        'std_education_education-num',
        'max_education_education-num',
        'mean_marital-status_age',
        'std_marital-status_age',
        'max_marital-status_age',
        'mean_marital-status_education-num',
        'std_marital-status_education-num',
        'max_marital-status_education-num',
        'mean_occupation_age',
        'std_occupation_age',
        'max_occupation_age',
        'mean_occupation_education-num',
        'std_occupation_education-num',
        'max_occupation_education-num',
        'mean_relationship_age',
        'std_relationship_age',
        'max_relationship_age',
        'mean_relationship_education-num',
        'std_relationship_education-num',
        'max_relationship_education-num',
        'mean_race_age',
        'std_race_age',
        'max_race_age',
        'mean_race_education-num',
        'std_race_education-num',
        'max_race_education-num',
        'mean_sex_age',
        'std_sex_age',
        'max_sex_age',
        'mean_sex_education-num',
        'std_sex_education-num',
        'max_sex_education-num',
        'workclass_diff_age',
        'workclass_diff_education-num',
        'education_diff_age',
        'education_diff_education-num',
        'marital-status_diff_age',
        'marital-status_diff_education-num',
        'occupation_diff_age',
        'occupation_diff_education-num',
        'relationship_diff_age',
        'relationship_diff_education-num',
        'race_diff_age',
        'race_diff_education-num',
        'sex_diff_age',
        'sex_diff_education-num',
        'freq_workclass',
        'freq_education',
        'freq_marital-status',
        'freq_occupation',
        'freq_relationship',
        'freq_race',
    ]
    # CV設定の読み込み
    cv_setting = get_cv_info()

    # run name設定の読み込み
    run_name = get_run_name(cv_setting, model_type="xgb")
    dir_name = MODEL_DIR_NAME + run_name + '/'

    # ログを吐くディレクトリを作成する
    # exist_check(MODEL_DIR_NAME, run_name)  # これは使わない、使うと実行が終わらない
    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # ファイルの設定を読み込む
    file_setting = get_file_info()

    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["hopt"] = "xgb_hopt"

    # xgbパラメータを設定する
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.1,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'min_child_weight': 1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 71,
        'num_round': 5000,
        "verbose": False,
        'early_stopping_rounds': 100,
    }

    # インスタンス生成
    runner = Runner(run_name, ModelXGB, features, params, file_setting, cv_setting, run_setting)

    # 今回の学習で使用した特徴量名を取得
    use_feature_name = runner.get_feature_name() 

    # 今回の学習で使用したパラメータを取得
    use_params = runner.get_params()

    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # 学習
    if cv_setting.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 予測
    else:
        runner.run_train_cv()  # 学習
        ModelXGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
        runner.run_predict_cv()  # 予測

    # submissionファイルの作成
    # 今回は、出力が確率なのでラベルに変換
    train_labels = pd.read_pickle(FEATURE_DIR_NAME + f'{file_setting.get("train_file_name")}')[run_setting.get("target")]
    xgb_train_probs = Util.load_df_pickle(dir_name + f'{run_name}-train_preds.pkl')
    xgb_probs = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    xgb_preds = get_label(train_labels, xgb_train_probs, xgb_probs)

    Submission.create_submission(run_name, dir_name, xgb_preds)  # submit作成

    

##### 学習・推論 LightGBM #############################################################

    features = features

    # CV設定の読み込み
    cv_setting = get_cv_info(random_state=86)

    # run nameの設定
    run_name = get_run_name(cv_setting, model_type="lgb")
    dir_name = MODEL_DIR_NAME + run_name + '/'

    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # ファイルの設定を読み込む
    file_setting = get_file_info()
    
    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["hopt"] = "lgb_hopt"
    run_setting["calc_shap"] = False

    # モデルのパラメータ
    params = {
      'boosting_type': 'gbdt',
      "objective": 'binary',
      "learning_rate": 0.01,
      "num_leaves": 31,
      'colsample_bytree': .5,
      "reg_lambda": 5,
      'random_state': 71,
      'num_boost_round': 5000,
      "verbose_eval": False,
      'early_stopping_rounds': 100,
    }

    runner = Runner(run_name, ModelLGB, features, params, file_setting, cv_setting, run_setting)

    # 今回の学習で使用した特徴量名を取得
    use_feature_name = runner.get_feature_name() 

    # 今回の学習で使用したパラメータを取得
    use_params = runner.get_params()

    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # 学習
    if cv_setting.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 予測
    else:
        runner.run_train_cv()  # 学習
        ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
        runner.run_predict_cv()  # 予測

    # submissionファイルの作成
    # 今回は,出力が確率なので,閾値の最適化後にラベル変換
    train_labels = pd.read_pickle(FEATURE_DIR_NAME + f'{file_setting.get("train_file_name")}')[run_setting.get("target")]
    lgb_train_probs = Util.load_df_pickle(dir_name + f'{run_name}-train_preds.pkl')
    lgb_probs = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    lgb_preds = get_label(train_labels, lgb_train_probs, lgb_probs)

    Submission.create_submission(run_name, dir_name, lgb_preds)  # submit作成


##### ニューラルネットワーク ###########################################################

    features = features

    # CV設定の読み込み
    cv_setting = get_cv_info(random_state=53)

    # run nameの設定
    run_name = get_run_name(cv_setting, model_type="nn")
    dir_name = MODEL_DIR_NAME + run_name + '/'

    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # ファイルの設定を読み込む
    file_setting = get_file_info()
    
    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["hopt"] = False
    run_setting["calc_shap"] =False

    # モデルのパラメータ
    params = {
        "num_classes": 1, 
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2,
        'batch_norm': 'before_act',
        "output_activation": "sigmoid",
        'optimizer': {'type': 'adam', 'lr': 0.000005},
        "loss": "binary_crossentropy", 
        "metrics": "accuracy", # カスタム評価関数も使える
        'batch_size': 64,
    }
    # params = {
    #     "num_classes": 1,
    #     "input_dropout": 0.1,
    #     "hidden_layers": 4.0,
    #     "hidden_units": 160.0,
    #     "hidden_activation": "relu",
    #     "hidden_dropout": 0.30000000000000004,
    #     "batch_norm": "no",
    #     "output_activation": "sigmoid",
    #     "optimizer": {
    #         "lr": 0.009838711682220185,
    #         "type": "sgd"
    #     },
    #     "loss": "binary_crossentropy",
    #     "metrics": "accuracy",
    #     "batch_size": 64.0
    # }

    runner = Runner(run_name, ModelNN, features, params, file_setting, cv_setting, run_setting)

    # 今回の学習で使用した特徴量名を取得
    use_feature_name = runner.get_feature_name() 

    # 今回の学習で使用したパラメータを取得
    use_params = runner.get_params()

    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # 学習
    if cv_setting.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 予測
    else:
        runner.run_train_cv()  # 学習
        ModelNN.plot_learning_curve(run_name)  # 学習曲線を描画
        runner.run_predict_cv()  # 予測

    # submissionファイルの作成
    # 今回は,出力が確率なので,閾値の最適化後にラベル変換
    train_labels = pd.read_pickle(FEATURE_DIR_NAME + f'{file_setting.get("train_file_name")}')[run_setting.get("target")]
    nn_train_probs = Util.load_df_pickle(dir_name + f'{run_name}-train_preds.pkl')
    nn_probs = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    nn_preds = get_label(train_labels, nn_train_probs, nn_probs)

    Submission.create_submission(run_name, dir_name, nn_preds)  # submit作成



##### アンサンブル ####################################################################

    run_name = get_run_name(cv_setting, "ensemble")

    # アンサンブル
    em_train_probs = xgb_train_probs*0.35 + lgb_train_probs*0.35 + nn_train_probs*0.3
    em_probs = xgb_probs*0.35 + lgb_probs*0.35 + nn_probs*0.3
    em_preds = get_label(train_labels, em_train_probs, em_probs)

    Submission.create_submission(run_name, dir_name, em_preds)  # submit作成
