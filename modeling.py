import os
import pandas as pd
import optuna
import numpy as np
import random
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from collections import Counter
from joblib import Parallel, delayed
from functools import partial
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.model_selection import train_test_split

from script.utils import save_df, load_features
from script.data_processing import save_f1_scores, preprocess_data
from script.config import LABEL_COLS, CAT_COLS

def objective(X, y, cat_cols, trial):
    """
    Optunaの目的関数
    """
    # RFE で取り出す特徴量の数を最適化する
    # n_features_to_select = trial.suggest_int('n_features_to_select', 1, X.shape[1])
    n_features_to_select = trial.suggest_int('n_features_to_select', 20, X.shape[1]//2)
    # データ型の確認とカテゴリカル列の特定
    categorical_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or col in cat_cols:
            categorical_features.append(col)
    
    # カテゴリカル特徴量のエンコーディング
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_processed = X.copy()
    if categorical_features:
        X_cat_encoded = pd.DataFrame(
            encoder.fit_transform(X[categorical_features]), 
            columns=categorical_features
        )
        X_processed[categorical_features] = X_cat_encoded

    # 数値特徴量のスケーリング
    numeric_features = [col for col in X.columns if col not in categorical_features]
    if numeric_features:
        scaler = StandardScaler()
        X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])

    # RFEによる特徴量選択
    clf = CatBoostClassifier(
        iterations=100,
        random_seed=42,
        logging_level='Silent'
    )
    rfe = RFE(estimator=clf, n_features_to_select=n_features_to_select)
    rfe.fit(X_processed, y)
    selected_indices = rfe.get_support(indices=True)
    selected_features = X_processed.columns[selected_indices]

    # 選択された特徴量を用いてモデルを評価
    X_selected = X_processed[selected_features]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_selected, y, cv=skf, method='predict')

    # 評価指標としてF1スコアを使用
    metric = f1_score(y, y_pred)
    
    # 特徴量をトライアルに保存
    trial.set_user_attr('selected_features', list(selected_features))
    
    return metric

def select_features_by_optuna(trains, output_dir):
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Add fixed seed to sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    
    for target_col in LABEL_COLS:
        if target_col not in trains.columns:
            continue
        print(f"\nTarget Variable: {target_col}")
        X = trains.drop(columns=LABEL_COLS)
        y = trains[target_col]

        f = partial(objective, X, y, CAT_COLS)

        # Use sampler with fixed seed
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        study.optimize(f, n_trials=20)
        
        selected_features = study.best_trial.user_attrs['selected_features']
        
        def save_features(features, save_dir, filename):
            """
            選択された特徴量リストをテキストファイルに保存する。
            """
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'w') as f:
                for feat in features:
                    f.write(f"{feat}\n")
        # 特徴量の保存
        save_features(selected_features, os.path.join(output_dir, target_col, 'optuna'), 'selected_features.txt')


def train_models(train_data, feature_dir, feature_importance_output_dir, game_id, method, param, unused_cols):
    """
    train_data全体で学習を行い、目的変数ごとの学習済みモデルを返す
    """
    # 不要列をドロップ
    train_data = train_data.drop(columns=unused_cols, errors='ignore').copy()
    model_dict = {}

    for target in LABEL_COLS:
        selected_features = load_features(feature_dir, target, method, param)
        if not selected_features:
            model_dict[target] = None
            continue
        # Remove unused columns from selected features
        selected_features = [col for col in selected_features if col not in unused_cols]
        
        feature_cols = [col for col in selected_features if col not in LABEL_COLS]
        cat_cols = [col for col in CAT_COLS if col in feature_cols]
        
        # データをトレーニングセットと検証セットに分割
        train_set, valid_set = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data[target])

        X_train_processed, y_train, X_valid_processed, y_valid, scaler, encoder, imputer = preprocess_data(
        train_set, valid_set, target, feature_cols, cat_cols, return_preprocessors=True
        )
        # 学習
        model_name = f"Train_{target}"
        _, feature_importance_df, model = fit_predict_model_return_model(
            X_train_processed, y_train, X_valid_processed, y_valid,
            model_name, feature_cols, cat_cols
        )
        model_dict[target] = (model, feature_cols, cat_cols, scaler, encoder, imputer)
        
        # Feature Importance
        os.makedirs(feature_importance_output_dir, exist_ok=True)
        target_name = target.replace('_future_label', '')
        csv_path = os.path.join(
            feature_importance_output_dir,
            f'{game_id}_{target_name}.csv'
        )
        feature_importance_df.to_csv(csv_path, index=False)
    return model_dict

def predict_with_trained_models(test_data, model_dict, game_id, subset_id,
                                proba_output_dir, f1_scores_output_path):
    """
    すでに学習済みのmodel_dictを使って、test_dataに対する予測を行う
    """
    test_data = test_data.copy()
    targets = ['concede_future_label', 'foul_future_label', 'fast_break_future_label']
    model_names = ['Concede', 'Foul', 'Fast Break']
    f1_scores = {}

    for target, model_name in zip(targets, model_names):
        model_info = model_dict.get(target)
        if not model_info:
            test_data[f'preds_proba_{target.replace("_future_label","")}'] = 0
            f1_scores[target] = 0
            continue

        model, feature_cols, cat_cols, scaler, encoder, imputer = model_info

        # 保存したスケーラーとエンコーダーを使用して前処理
        _, _, X_test_processed, y_test = preprocess_data(
            None, test_data, target_col=target,
            all_feature_cols=feature_cols, cat_cols=cat_cols,
            return_preprocessors=False,  # Add explicit parameter
            scaler=scaler, encoder=encoder, imputer=imputer
        )

        preds_proba = model.predict_proba(X_test_processed)[:, 1]
        preds_class = (preds_proba >= 0.5).astype(int)
        f1 = f1_score(y_test, preds_class, zero_division=0)
        f1_scores[target] = f1
        test_data[f'preds_proba_{target.replace("_future_label","")}'] = preds_proba

    # スコア等の保存
    save_f1_scores(
        f1_scores_output_path,
        game_id,
        subset_id,
        f1_scores.get('concede_future_label', 0),
        f1_scores.get('foul_future_label', 0),
        f1_scores.get('fast_break_future_label', 0)
    )

    # 予測結果をJSON保存
    save_df(test_data, proba_output_dir, game_id, subset_id)
    return test_data

def fit_predict_model_return_model(X_train, y_train, X_valid, y_valid, model_name, feature_names, cat_cols):
    """
    学習後、モデルオブジェクトを返すための関数
    """
    
    """ model = CatBoostClassifier(
        custom_loss=['Accuracy'],
        random_seed=42,
        verbose=0
    ) """
    model = CatBoostClassifier( # custom_loss=['Accuracy']を削除
        eval_metric='F1',
        iterations=100, # <- iterationsの設定を追加
        random_seed=42,
        logging_level='Silent',
        cat_features=cat_cols # <- cat_featuresの設定を追加
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols) if X_valid is not None and y_valid is not None else None

    # model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=10, use_best_model=True)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True) # <- early_stoppingを削除
    feature_importances = model.get_feature_importance(train_pool)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print(f"{model_name} trained.")
    return None, feature_importance_df, model