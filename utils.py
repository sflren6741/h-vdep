import datetime
import numpy as np
import os
import pandas as pd

def format_timedelta(timedelta):
    total_sec = timedelta.total_seconds()
    hours = total_sec // 3600
    remain = total_sec - (hours * 3600)
    minutes = remain // 60
    seconds = remain - (minutes * 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

def save_df(df, output_dir, game_id, subset_id):
    df.reset_index(drop=True, inplace=True)
    if not os.path.exists(output_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(output_dir)
    df.to_json(os.path.join(output_dir, f'{game_id}_{subset_id}.json'))

def load_features(feature_dir, target, method, param=None):
    """
    特徴量をファイルから読み込む
    """
    if method in ['chi2', 'anova', 'mutual_info', 'rfe']:
        feature_file = os.path.join(feature_dir, target, method, f'top_{param}_features.txt')
    elif method == 'correlation_based':
        feature_file = os.path.join(feature_dir, target, method, f'threshold_{param}.txt')
    else:
        feature_file = os.path.join(feature_dir, target, method, 'selected_features.txt')

    if not os.path.exists(feature_file):
        print(f"Feature file not found: {feature_file}")
        return []

    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f]
    return features