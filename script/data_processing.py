import os
import re

import numpy as np
import pandas as pd
import h5py
import csv
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field
from typing import List, Dict

from script.utils import format_timedelta
from script.config import LABEL_COLS, CAT_COLS, DATE

def calc_fast_break_success_rate(proba):
    shot_time = None
    possession_start_time = None
    predict_event_num_ = 3
    fast_break_num = 0
    fast_break_success = 0
    # 速攻のラベル付け
    # ポゼッションチェンジから15秒以内にシュートがあるかを判定
    for i in range(proba.shape[0]):
        fast_break_success_flag = 0
        if proba.at[i, 'event_type'] == 'possession_change':
            possession_start_time = (proba.at[i, 'start_time']-1)/30
            shot_time = None
        if (proba.at[i, 'event_type'] == 'shot') and (shot_time == None):
            shot_time = (proba.at[i, 'start_time']-1)/30
            for j in range(i+1, i+1+predict_event_num_):
                if j < proba.shape[0]:
                    if proba.at[j, 'event_type'] == 'goal':
                        fast_break_success_flag += 1
                        break
            if (possession_start_time != None) and (shot_time != None):
                # ポゼッションチェンジから15秒以内にシュートがある場合、速攻と判定
                if (shot_time - possession_start_time) < 15.0:
                    fast_break_num += 1
                    fast_break_success += fast_break_success_flag
    # print(f"fast_break_num : {fast_break_num}")
    # print(f"fast_break_success : {fast_break_success}")
    return fast_break_num, fast_break_success

def calc_after_foul_concede_rate(proba):
    preds_proba_concede = []
    for i in range(proba.shape[0] - 1):  # 最後の行は次の行がないため除外
        if proba.at[i, 'event_type'] == 'foul_free_throw' and proba.at[i + 1, 'event_type'] == 'free_throw':
            preds_proba_concede.append(proba.at[i + 1, 'preds_proba_concede'])
    return preds_proba_concede

def linear_regression(preds_proba_concede, after_foul_concede_rate_mean):
    if preds_proba_concede <= after_foul_concede_rate_mean:
        y = 0
    else:
        y = preds_proba_concede - after_foul_concede_rate_mean
    return y

def load_after_foul_concede_rate_mean(filepath):
    """
    指定されたテキストファイルからafter_foul_concede_rate_meanを取得する関数

    Parameters:
    - filepath: テキストファイルのパス

    Returns:
    - after_foul_concede_rate_mean: 取得した値
    """
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("after_foul_concede_rate_mean:"):
                return float(line.split(":")[1].strip())
    raise ValueError("after_foul_concede_rate_mean not found in the file")

def calc_vdep_(proba, fast_break_weight, after_foul_concede_rate_mean, game_id, subset_id, output_dir):
    print(f"{game_id}_{subset_id}")
    for index, row in proba.iterrows():
        # logistic関数のパラメータを設定(a=11)
        foul_weight = linear_regression(row['preds_proba_concede'], after_foul_concede_rate_mean)
        vdep = fast_break_weight * row['preds_proba_fast_break'] + foul_weight * row['preds_proba_foul'] - row['preds_proba_concede']
        proba.at[index, 'vdep'] = vdep

    if not os.path.exists(output_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
    proba.to_json(os.path.join(output_dir, f'{game_id}_{subset_id}.json'))
    
def calc_vdep(proba_path, vdep_output_dir, weights_output_path):
    # 速攻時の得点期待値の平均を計算する関数
    proba_ = os.listdir(proba_path)
    fast_break_num = 0
    fast_break_success = 0
    # Loop through labeled features to calculate vdep
    for i in tqdm(range(len(proba_))):
        with open(os.path.join(proba_path, proba_[i]), 'rb') as f:
            proba = pd.read_json(f)
        game_id = re.split('[_.]', proba_[i])[0]
        subset_id = re.split('[_.]', proba_[i])[1]
        fast_break_num_, fast_break_success_ = calc_fast_break_success_rate(proba)
        fast_break_num += fast_break_num_
        fast_break_success += fast_break_success_
    fast_break_weight = fast_break_success / fast_break_num
    print(f"fast_break_num : {fast_break_num}")
    print(f"fast_break_success : {fast_break_success}")
    print(f"fast_break_weight : {fast_break_weight}")
    
    after_foul_concede_rate = []
    for i in tqdm(range(len(proba_))):
        with open(os.path.join(proba_path, proba_[i]), 'rb') as f:
            proba = pd.read_json(f)
        after_foul_concede_rate.extend(calc_after_foul_concede_rate(proba))
    after_foul_concede_rate_mean = np.mean(after_foul_concede_rate)
    # print(f"after_foul_concede_rate : {after_foul_concede_rate}")
    print(f"after_foul_concede_rate_mean : {after_foul_concede_rate_mean}")
    
    # ディレクトリ部分を抽出
    weights_output_dir = os.path.dirname(weights_output_path)

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(weights_output_dir):
        os.makedirs(weights_output_dir, exist_ok=True)
        
    # 重みづけの値をテキストファイルに保存
    with open(weights_output_path, 'w') as f:
        f.write(f"fast_break_weight: {fast_break_weight}\n")
        f.write(f"after_foul_concede_rate_mean: {after_foul_concede_rate_mean}\n")
    print(f"Weights saved to {weights_output_path}")

    for i in tqdm(range(len(proba_))):
        with open(os.path.join(proba_path, proba_[i]), 'rb') as f:
            proba = pd.read_json(f)
        game_id = re.split('[_.]', proba_[i])[0]
        subset_id = re.split('[_.]', proba_[i])[1]
        calc_vdep_(proba, fast_break_weight, after_foul_concede_rate_mean, game_id, subset_id, vdep_output_dir)
        
def calc_scores(vdep, game_id, subset_id, after_foul_concede_rate_mean):
    print(f"{game_id}_{subset_id}")
    vdep_team_a = []
    vdep_team_b = []
    concede_team_a = 0
    concede_team_b = 0
    foul_team_a = 0
    foul_team_b = 0
    fast_break_team_a = 0
    fast_break_team_b = 0
    concede_prob_team_a = []
    concede_prob_team_b = []
    foul_prob_team_a = []
    foul_prob_team_b = []
    fast_break_prob_team_a = []
    fast_break_prob_team_b = []
    transformed_foul_prob_team_a = []
    transformed_foul_prob_team_b = []

    for index, row in vdep.iterrows():

        if row['possession_team'] == 0:
            # チームBが守備の場合
            vdep_team_b.append(row['vdep'])
            concede_prob_team_b.append(row['preds_proba_concede'])
            foul_prob_team_b.append(row['preds_proba_foul'])
            fast_break_prob_team_b.append(row['preds_proba_fast_break'])
            transformed_foul_prob_team_b.append(linear_regression(row['preds_proba_foul'], after_foul_concede_rate_mean))
            if (row['event_type'] == 'shot') and (row['result'] == 'successful'):
                concede_team_b += 1
            elif (row['event_type'] == 'foul') and (row['result'] == 'successful'):
                foul_team_b += 1
            elif (row['event_type'] == 'fast_break') and (row['result'] == 'successful'):
                fast_break_team_b += 1
        elif row['possession_team'] == 1:
            # チームAが守備の場合
            vdep_team_a.append(row['vdep'])
            concede_prob_team_a.append(row['preds_proba_concede'])
            foul_prob_team_a.append(row['preds_proba_foul'])
            fast_break_prob_team_a.append(row['preds_proba_fast_break'])
            transformed_foul_prob_team_a.append(linear_regression(row['preds_proba_foul'], after_foul_concede_rate_mean))
            if (row['event_type'] == 'shot') and (row['result'] == 'successful'):
                concede_team_a += 1
            elif (row['event_type'] == 'foul') and (row['result'] == 'successful'):
                foul_team_a += 1
            elif (row['event_type'] == 'fast_break') and (row['result'] == 'successful'):
                fast_break_team_a += 1

    vdep_mean_team_a = np.mean(vdep_team_a)
    vdep_mean_team_b = np.mean(vdep_team_b)
    concede_prob_mean_team_a = np.mean(concede_prob_team_a)
    concede_prob_mean_team_b = np.mean(concede_prob_team_b)
    foul_prob_mean_team_a = np.mean(foul_prob_team_a)
    foul_prob_mean_team_b = np.mean(foul_prob_team_b)
    fast_break_prob_mean_team_a = np.mean(fast_break_prob_team_a)
    fast_break_prob_mean_team_b = np.mean(fast_break_prob_team_b)
    transformed_foul_prob_mean_team_a = np.mean(transformed_foul_prob_team_a)
    transformed_foul_prob_mean_team_b = np.mean(transformed_foul_prob_team_b)

    return (vdep_mean_team_a, vdep_mean_team_b, concede_team_a, concede_team_b, 
            foul_team_a, foul_team_b, fast_break_team_a, fast_break_team_b,
            concede_prob_mean_team_a, concede_prob_mean_team_b, 
            foul_prob_mean_team_a, foul_prob_mean_team_b, 
            fast_break_prob_mean_team_a, fast_break_prob_mean_team_b,
            transformed_foul_prob_mean_team_a, transformed_foul_prob_mean_team_b)


# Calculate the number of missing frames in the position data.
def calc_average_time():
    spadl_path = "./data/spadl"
    spadls = os.listdir(spadl_path)
    shot_times = []
    for spadl in spadls:
        #print(spadl)
        with open(os.path.join(spadl_path, spadl), 'rb') as f:
            df = pd.read_json(f)
        if spadl == "spadl_ec7a6a_01-40-00.json":
            print("possession_change")
            print(df.event_type_possession_change)
            print("shot")
            print(df.event_type_shot)
            print("start_time")
            print(df.start_time)
            
            
        # df = pd.read_json(os.path.join(spadl_path, spadl), lines=True)
        #print(spadl)
        possession_start_time = None
        shot_time = None
        for index, event in df.iterrows():
            #print(index)
            # print(event)
            
            if event.event_type_possession_change == 1:
                possession_start_time = (event.start_time-1)/30
                shot_time = None
                
            if (event.event_type_shot == 1) and (shot_time == None):
                shot_time = (event.start_time-1)/30
                if (possession_start_time != None) and (shot_time != None):
                    print(f"game_id:{spadl}")
                    print(f"possession_start_time:{format_timedelta(datetime.timedelta(seconds=possession_start_time))}")
                    print(f"shot_time:{format_timedelta(datetime.timedelta(seconds=shot_time))}")
                    print(f"time:{shot_time - possession_start_time:.2f}")
                    print(f"possession_start_frame:{int(possession_start_time*30)+1}")
                    print(f"shot_frame:{int(shot_time*30)+1}")
                    print("------------------------------")
                    shot_times.append(shot_time - possession_start_time)
                    
    #print(np.round(shot_times, 2))
    #print(f"shot_num:{len(shot_times)}")
    ft, n_bins = frequency_table(shot_times, stur=False, b=30)
    print(ft)
    fig, ax = plt.subplots(figsize=(12,4))
    plt.tick_params(labelsize=18)
    plt.xticks(np.arange(0, 100, step=10))
    plt.tight_layout()
    plt.xlabel("Time[s]")
    ax.hist(shot_times, bins=n_bins, ec='black', color='#005AFF')
    save_dir = f'./data/fig/average_time.png'
    fig.savefig(save_dir)
    
# Calculate average time from possession change to scoring.
def calc_average_DF_position():
    spadl_path = "./data/spadl"
    spadls = os.listdir(spadl_path)
    average_DF_positions_x = []
    shot_times = []
    h5_path = "./data/eigd/handball/positions/"
    positions = os.listdir(h5_path)
    for position in positions:
        with h5py.File(os.path.join(h5_path, position), 'r') as h5:
            print(f"len(h5['balls']) : {len(h5['balls'])}")
            #for i in range(len(h5['balls'])):
    for spadl in spadls:
        #print(spadl)
        with open(os.path.join(spadl_path, spadl), 'rb') as f:
            df = pd.read_json(f)
            
        
        # df = pd.read_json(os.path.join(spadl_path, spadl), lines=True)
        #print(spadl)
        possession_start_time = None
        shot_time = None
        for index, event in df.iterrows():
            #print(index)
            # print(event)
            
            if event.event_type_possession_change == 1:
                possession_start_time = (event.start_time-1)/30
                shot_time = None
                
            if (event.event_type_shot == 1) and (shot_time == None):
                shot_time = (event.start_time-1)/30
                if (possession_start_time != None) and (shot_time != None):
                    print(f"game_id:{spadl}")
                    print(f"possession_start_time:{int((possession_start_time+1)*30)}")
                    print(f"shot_time:{int((shot_time+1)*30)}")
                    print(f"time:{shot_time - possession_start_time}")
                    shot_times.append(shot_time - possession_start_time)
                    
    print(shot_times)
    print(f"shot_num:{len(shot_times)}")
    ft, n_bins = frequency_table(shot_times)
    fig, ax = plt.subplots(figsize=(12,4))
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    ax.hist(shot_times, bins=n_bins, ec='black', color='#005AFF')
    save_dir = f'./data/fig/average_time.png'
    fig.savefig(save_dir)

def calc_missing_frame(h5_paths, h5_dir):
    for h5_path in h5_paths:
        print(f"h5_path:{h5_path}")
        total_missing_frame_ball = 0
        total_missing_frame_team_a = 0
        total_missing_frame_team_b = 0
        missing_frame_ball_start = None
        missing_frame_ball_end = None
        missing_frame_team_a_start = None
        missing_frame_team_a_end = None
        missing_frame_team_b_start = None
        missing_frame_team_b_end = None

        with h5py.File(h5_dir + h5_path, "r") as h5:
            print(f"total frame:{len(h5['balls'])}")
            for i in range(0, 9000):
                if np.all(np.isnan(h5['balls'][i])):
                    total_missing_frame_ball += 1
                    if missing_frame_ball_start == None:
                        missing_frame_ball_start = datetime.timedelta(seconds=i/30)
                else:
                    if missing_frame_ball_start != None:
                        missing_frame_ball_end = datetime.timedelta(seconds=(i-1)/30)
                        print("ball : {:s}-{:s} is missing".format(format_timedelta(missing_frame_ball_start), format_timedelta(missing_frame_ball_end)))
                        missing_frame_ball_start = None
                        missing_frame_ball_end = None

                if np.all(np.isnan(h5['team_a'][i])):
                    total_missing_frame_team_a += 1
                    if missing_frame_team_a_start == None:
                        missing_frame_team_a_start = datetime.timedelta(seconds=i/30)
                else:
                    if missing_frame_team_a_start != None:
                        missing_frame_team_a_end = datetime.timedelta(seconds=(i-1)/30)
                        print("team_a : {:s}-{:s} is missing".format(format_timedelta(missing_frame_team_a_start), format_timedelta(missing_frame_team_a_end)))
                        missing_frame_team_a_start = None
                        missing_frame_team_a_end = None

                if np.all(np.isnan(h5['team_b'][i])):
                    total_missing_frame_team_b += 1
                    if missing_frame_team_b_start == None:
                        missing_frame_team_b_start = datetime.timedelta(seconds=i/30)
                else:
                    if missing_frame_team_b_start != None:
                        missing_frame_team_b_end = datetime.timedelta(seconds=(i-1)/30)
                        print("team_b : {:s}-{:s} is missing".format(format_timedelta(missing_frame_team_b_start), format_timedelta(missing_frame_team_b_end)))
                        missing_frame_team_b_start = None
                        missing_frame_team_b_end = None

            print(f"total_missing_frame_ball:{total_missing_frame_ball}")
            print(f"total_missing_frame_team_a:{total_missing_frame_team_a}")
            print(f"total_missing_frame_team_b:{total_missing_frame_team_b}")

def calc_and_plot_velocity(h5_file_path: str, player_id: int, team: str, output_file: str):
    # h5ファイルを開く
    with h5py.File(h5_file_path, "r") as h5:
        if team not in h5:
            print(f"Team '{team}' not found in {h5_file_path}")
            return
        
        team_data = h5[team]
        if len(team_data) == 0:
            print(f"No data for '{team}' in {h5_file_path}")
            return
        
        velocities = []
        for i in range(1, len(team_data)):
            # 現在と前のフレームの座標を取得
            if player_id >= len(team_data[i]) or player_id >= len(team_data[i - 1]):
                velocities.append(np.nan)
                continue
            
            curr_pos = team_data[i][player_id]
            prev_pos = team_data[i - 1][player_id]
            
            if np.any(np.isnan(curr_pos)) or np.any(np.isnan(prev_pos)):
                velocities.append(np.nan)
                continue
            
            # 速度を計算
            delta_pos = np.array(curr_pos) - np.array(prev_pos)
            velocity = np.linalg.norm(delta_pos)
            velocities.append(velocity)
        
        # 時間軸を作成
        time_frames = list(range(len(velocities)))
    
    # グラフの描画設定
    # 目標のピクセルサイズとdpi
    width_px = 9600
    height_px = 720
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 余白の調整

    # 速度プロット
    ax.plot(time_frames, velocities, color='blue', lw=2, label="velocity")
    
    # 軸や目盛りの設定
    ax.set_xlim(0, len(velocities))
    ax.yaxis.set_visible(False)  # y軸の目盛りを非表示にする
    ax.tick_params(which='both', direction='in', pad=10)
    ax.grid(True)

    # 出力ディレクトリが存在しない場合、作成
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)

def frequency_table(data, stur=True, b=10):
    data_len = len(data)
    if stur is True:
        b = round(1 + np.log2(data_len))
        hist, bins = np.histogram(data, bins=b) 
    else:
        hist, bins = np.histogram(data, bins=b)
    df = pd.DataFrame({
        '以上': bins[:-1],
        '以下': bins[1:],
        '階級値': (bins[:-1]+bins[1:])/2,
        '度数': hist
    })
    df['相対度数'] = df['度数'] / data_len
    df['累積度数'] = np.cumsum(df['度数'])
    df['累積相対度数'] = np.cumsum(df['相対度数'])
    return df, b

""" def preprocess_data(train, test, target_col, all_feature_cols, cat_cols):
    
    データを前処理し、数値特徴量のスケーリング、カテゴリカル特徴量のエンコーディング、
    欠損値の処理、SMOTENCによるクラス不均衡への対応を行います。

    Parameters:
    - train: トレーニングデータのDataFrame
    - test: テストデータのDataFrame
    - target_col: 予測するターゲット列の名前
    - all_feature_cols: 使用する全ての特徴量のリスト
    - cat_cols: カテゴリカル特徴量のリスト

    Returns:
    - X_resampled: 前処理およびリサンプリングされたトレーニング特徴量
    - y_resampled: リサンプリングされたトレーニングラベル
    - X_test_processed: 前処理されたテスト特徴量
    - y_test: テストラベル
   
    # print("Before preprocessing:")
    # print("train cols: ", train.columns.tolist())
    # print("test cols: ", test.columns.tolist())
    # 数値列のリストを作成
    num_cols = [col for col in all_feature_cols if col not in cat_cols]

    # 数値特徴量のスケーリング
    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(sc.fit_transform(train[num_cols]), columns=num_cols)
    X_test_scaled = pd.DataFrame(sc.transform(test[num_cols]), columns=num_cols)

    # カテゴリカル特徴量のエンコーディング
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat = pd.DataFrame(encoder.fit_transform(train[cat_cols]), columns=cat_cols)
    test_cat = pd.DataFrame(encoder.transform(test[cat_cols]), columns=cat_cols)
    train_cat = train_cat.astype(int)
    test_cat = test_cat.astype(int)

    # 数値特徴量とカテゴリカル特徴量を結合
    X_train_processed = pd.concat(
        [train_cat.reset_index(drop=True), X_train_scaled.reset_index(drop=True)],
        axis=1
    )
    X_test_processed = pd.concat(
        [test_cat.reset_index(drop=True), X_test_scaled.reset_index(drop=True)],
        axis=1
    )

    # 欠損値の処理
    if X_train_processed.isnull().values.any() or X_test_processed.isnull().values.any():
        # print("NaN values detected in processed features.")
        imputer = SimpleImputer(strategy='constant', fill_value=-1)
        X_train_processed = pd.DataFrame(
            imputer.fit_transform(X_train_processed),
            columns=X_train_processed.columns
        )
        X_test_processed = pd.DataFrame(
            imputer.transform(X_test_processed),
            columns=X_test_processed.columns
        )

    # SMOTENCのためのカテゴリカル特徴量のインデックスを取得
    categorical_indices = [X_train_processed.columns.get_loc(col) for col in cat_cols]

    # カテゴリカル特徴量が存在する場合のみSMOTENCを適用
    if categorical_indices:
        # SMOTENCによるクラス不均衡への対応
        smote_nc = SMOTENC(
            categorical_features=categorical_indices,
            random_state=42,
            k_neighbors=NearestNeighbors(n_jobs=-1)
        )
        X_resampled, y_resampled = smote_nc.fit_resample(
            X_train_processed,
            train[target_col]
        )
    else:
        X_resampled, y_resampled = X_train_processed, train[target_col]

    # カテゴリカル列のデータ型を整数型に変換
    X_resampled = pd.DataFrame(X_resampled, columns=X_train_processed.columns)
    X_resampled[cat_cols] = X_resampled[cat_cols].astype(int)
    X_test_processed[cat_cols] = X_test_processed[cat_cols].astype(int)

    return X_resampled, y_resampled, X_test_processed, test[target_col] """
    
def preprocess_data(train, test, target_col, all_feature_cols, cat_cols, return_preprocessors=False, scaler=None, encoder=None, imputer=None):
    """
    データを前処理し、数値特徴量のスケーリング、カテゴリカル特徴量のエンコーディング、
    欠損値の処理、SMOTENCによるクラス不均衡への対応を行います。

    Parameters:
    - train: トレーニングデータのDataFrame
    - test: テストデータのDataFrame
    - target_col: 予測するターゲット列の名前
    - all_feature_cols: 使用する全ての特徴量のリスト
    - cat_cols: カテゴリカル特徴量のリスト

    Returns:
    - X_resampled: 前処理およびリサンプリングされたトレーニング特徴量
    - y_resampled: リサンプリングされたトレーニングラベル
    - X_test_processed: 前処理されたテスト特徴量
    - y_test: テストラベル
    """
    # 数値列のリストを作成
    num_cols = [col for col in all_feature_cols if col not in cat_cols]

    if train is not None:
        # トレーニングデータの前処理
        sc = StandardScaler()
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # 数値特徴量のスケーリング
        X_train_scaled = pd.DataFrame(sc.fit_transform(train[num_cols]), columns=num_cols)
        X_test_scaled = pd.DataFrame(sc.transform(test[num_cols]), columns=num_cols) if test is not None else None

        # カテゴリカル特徴量のエンコーディング
        train_cat = pd.DataFrame(enc.fit_transform(train[cat_cols]), columns=cat_cols)
        test_cat = pd.DataFrame(enc.transform(test[cat_cols]), columns=cat_cols) if test is not None else None
        train_cat = train_cat.astype(int)
        if test_cat is not None:
            test_cat = test_cat.astype(int)

        # 数値特徴量とカテゴリカル特徴量を結合
        X_train_processed = pd.concat(
            [train_cat.reset_index(drop=True), X_train_scaled.reset_index(drop=True)],
            axis=1
        )
        X_test_processed = pd.concat(
            [test_cat.reset_index(drop=True), X_test_scaled.reset_index(drop=True)],
            axis=1
        ) if test is not None else None

        # 欠損値の処理
        if X_train_processed.isnull().values.any() or (X_test_processed is not None and X_test_processed.isnull().values.any()):
            imputer = SimpleImputer(strategy='constant', fill_value=-1)
            X_train_processed = pd.DataFrame(
                imputer.fit_transform(X_train_processed),
                columns=X_train_processed.columns
            )
            if X_test_processed is not None:
                X_test_processed = pd.DataFrame(
                    imputer.transform(X_test_processed),
                    columns=X_test_processed.columns
                )
        
        # クラス分布を表示（SMOTENC適用前）
        print(f"\nClass distribution before SMOTENC ({target_col}):")
        print(pd.Series(train[target_col]).value_counts())

        if cat_cols:
            # カテゴリカル特徴量のインデックスを取得
            categorical_indices = [X_train_processed.columns.get_loc(col) for col in cat_cols]
            
            # SMOTENCを適用（カテゴリカル特徴量の位置を指定）
            smote_nc = SMOTENC(
                categorical_features=categorical_indices,
                random_state=42,
                k_neighbors=NearestNeighbors(n_jobs=-1)
            )
            X_resampled, y_resampled = smote_nc.fit_resample(
                X_train_processed,
                train[target_col]
            )
        else:
            # カテゴリカル特徴量が存在しない場合は通常のSMOTEを適用
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(
                X_train_processed,
                train[target_col]
            )

        # クラス分布を表示（SMOTENC適用後）
        print(f"\nClass distribution after SMOTENC ({target_col}):")
        print(pd.Series(y_resampled).value_counts())

        # カテゴリカル列のデータ型を整数型に変換
        X_resampled = pd.DataFrame(X_resampled, columns=X_train_processed.columns)
        X_resampled[cat_cols] = X_resampled[cat_cols].astype(int)
        if X_test_processed is not None:
            X_test_processed[cat_cols] = X_test_processed[cat_cols].astype(int)

        if return_preprocessors:
            return X_resampled, y_resampled, X_test_processed, (test[target_col] if test is not None else None), sc, enc, imputer
        return X_resampled, y_resampled, X_test_processed, test[target_col] if test is not None else None
        # return X_resampled, y_resampled, X_test_processed, test[target_col] if test is not None else None

    else:
        # テストデータのみの前処理
        X_test_scaled = pd.DataFrame(scaler.transform(test[num_cols]), columns=num_cols)

        test_cat = pd.DataFrame(encoder.transform(test[cat_cols]), columns=cat_cols)
        test_cat = test_cat.astype(int)

        X_test_processed = pd.concat(
            [test_cat.reset_index(drop=True), X_test_scaled.reset_index(drop=True)],
            axis=1
        )

        if X_test_processed.isnull().values.any():
            X_test_processed = pd.DataFrame(
                imputer.transform(X_test_processed),
                columns=X_test_processed.columns
            )

        X_test_processed[cat_cols] = X_test_processed[cat_cols].astype(int)

        return None, None, X_test_processed, test[target_col]

def save_f1_scores(filename, game_id, subset_id, f1_concede, f1_foul, f1_fast_break):
    data = {
        'game_id': game_id,
        'subset_id': subset_id,
        'f1_concede': f1_concede,
        'f1_foul': f1_foul,
        'f1_fast_break': f1_fast_break
    }
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['game_id', 'subset_id', 'f1_concede', 'f1_foul', 'f1_fast_break']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def logistic(x, x0, a=1, k=1):
    y = k / (1 + np.exp(-a * (x - x0)))
    return y

@dataclass
class TeamScores:
    vdeps: List[float] = field(default_factory=list)
    concedes: List[float] = field(default_factory=list)
    fouls: List[float] = field(default_factory=list)
    concede_probs: List[float] = field(default_factory=list)
    foul_probs: List[float] = field(default_factory=list)
    fast_break_probs: List[float] = field(default_factory=list)
    transformed_foul_probs: List[float] = field(default_factory=list)

    def get_mean_scores(self) -> Dict[str, float]:
        return {
            'vdep': np.mean(self.vdeps),
            'concede': np.mean(self.concedes),
            'foul': np.mean(self.fouls),
            'concede_prob': np.mean(self.concede_probs),
            'foul_prob': np.mean(self.foul_probs),
            'fast_break_prob': np.mean(self.fast_break_probs),
            'transformed_foul_prob': np.mean(self.transformed_foul_probs)
        }

def process_game_data(scores_data: Dict[str, TeamScores], game_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Process single game data and return scores dictionary"""
    return {
        game_id: {
            'team_a': scores_data['team_a'].get_mean_scores(),
            'team_b': scores_data['team_b'].get_mean_scores()
        }
    }
