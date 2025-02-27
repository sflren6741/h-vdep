import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import script.convert_to_actions as cta
from script.data_processing import calc_scores, calc_and_plot_velocity, calc_vdep, TeamScores, process_game_data, load_after_foul_concede_rate_mean
from script.utils import load_features
from script.visualization import (
    animate_positions, plot_feature_importance, plot_f1_scores,
    plot_vdep_relation, plot_vdep_relation_season, plot_vdep_by_event,
    animate_vdep_scroll, concat_videos_vertically, concat_videos_horizontally,
    overlay_videos, plot_vdep_by_event_with_event_type
)
from script.modeling import select_features_by_optuna, train_models, predict_with_trained_models
from script.config import DATE, PREDICT_EVENT_NUM, UNUSED_COLS

# === Step Execution Flags ===

# Main steps
step1 = False  # Convert event annotation data to SPADL format
step2 = False  # Convert event data to features
step3 = False  # Label features
step4 = False  # Feature selection
step5 = False  # Calculate probability
step6 = False # Calculate VDEP
step7 = False # Plot F1 scores
step8 = True # Plot VDEP relation
step9 = False # Plot VDEP by event
step10 = False  # Create VDEP scroll animation
step11 = False  # Concatenate videos vertically

# Additional steps
step12 = False  # Create player position animations
step13 = False  # Plot player velocity
step14 = False  # Concatenate game and velocity videos
step15 = False  # Overlay position animations on game videos
step16 = False  # Concatenate videos horizontally
step17 = False  # Plot feature importance

# === File Paths and Parameters for Each Step ===

# Common paths
h5_dir = "./data/eigd/handball/positions/"
json_path = "./data/eigd/handball/event_annotations_20241203.jsonl"

# Step 1: Convert event annotation data to SPADL format
spadl_output_dir = f'./data/json_files/spadl/spadl_{DATE}'

# Step 2: Convert event data to features
spadl_dir = './data/json_files/spadl/spadl_20241204'
features_output_dir = f'data/json_files/features/{DATE}'

# Step 3: Label features
features_dir = f'data/json_files/features/20250122'
labeled_features_output_dir = f'data/json_files/labeled_features/{DATE}_predict_{PREDICT_EVENT_NUM}events'

# Step 4: Feature selection
# labeled_features_dir = f'data/json_files/labeled_features/{DATE}_predict_{PREDICT_EVENT_NUM}events'
labeled_features_dir = f'data/json_files/labeled_features/20250127_predict_{PREDICT_EVENT_NUM}events'
selected_features_output_dir = f'./data/text_files/features/{DATE}_predict_{PREDICT_EVENT_NUM}events'

# Step 5: Calculate Probability
# selected_features_dir = f'data/text_files/features/{DATE}_predict_{PREDICT_EVENT_NUM}events'
selected_features_dir = f'data/text_files/features/20250127_predict_{PREDICT_EVENT_NUM}events'
proba_output_dir = f'data/json_files/proba/{DATE}_predict_{PREDICT_EVENT_NUM}events'
f1_scores_output_path = f'./data/csv_files/f1_scores/{DATE}_predict_{PREDICT_EVENT_NUM}events.csv'
feature_importance_output_dir = f"data/csv_files/feature_importances/{DATE}_predict_{PREDICT_EVENT_NUM}events"

# Step 6: Calculate VDEP
proba_path = f'data/json_files/proba/{DATE}_predict_{PREDICT_EVENT_NUM}events'
vdep_output_dir = f'data/json_files/vdep/{DATE}_predict_{PREDICT_EVENT_NUM}events'
weights_output_path = f'data/text_files/weights/weights_{DATE}_predict_{PREDICT_EVENT_NUM}events.txt'

# Step 7: Plot F1 scores
f1_scores_path = f'./data/csv_files/f1_scores/{DATE}_predict_{PREDICT_EVENT_NUM}events.csv'
# f1_plot_output_dir = f"./data/fig/f1_scores/f1_score_{DATE}_predict_{PREDICT_EVENT_NUM}events.png"
f1_plot_output_dir = f"./data/fig/f1_scores/{DATE}_predict_{PREDICT_EVENT_NUM}events"

# Step 8: Plot VDEP relation
vdep_path = f"data/json_files/vdep/{DATE}_predict_{PREDICT_EVENT_NUM}events"
# vdep_concede_relation_output_dir = f"./data/fig/vdep_concede_relation/{DATE}_predict_{PREDICT_EVENT_NUM}events"
vdep_concede_relation_output_dir = f"./data/fig/vdep_concede_relation/20250205_predict_{PREDICT_EVENT_NUM}events"
season_data_path = "data/csv_files/teams_data.csv"
# vdep_concede_relation_season_output_dir = f"./data/fig/vdep_concede_relation_season/{DATE}_predict_{PREDICT_EVENT_NUM}events"
vdep_concede_relation_season_output_dir = f"./data/fig/vdep_concede_relation_season/20250205_predict_{PREDICT_EVENT_NUM}events"
weights_path = f"data/text_files/weights/weights_{DATE}_predict_{PREDICT_EVENT_NUM}events.txt"

# Step 9: Plot VDEP by event
vdep_event_plot_with_event_type_output_dir = f'data/fig/vdep_by_event_with_event_type/{DATE}_predict_{PREDICT_EVENT_NUM}events'
vdep_event_plot_output_dir = f'data/fig/vdep_by_event/{DATE}_predict_{PREDICT_EVENT_NUM}events'

# Step 10: Create VDEP scroll animation
vdep_event_images_dir = f"./data/fig/vdep_by_event/{DATE}_predict_{PREDICT_EVENT_NUM}events"
legend_image_path = "data/fig/legend_image_20250122.png"
vdep_scroll_output_dir = f"data/videos/vdep_scroll/{DATE}_predict_{PREDICT_EVENT_NUM}events"

# Step 11: Concatenate videos
# video_dirs = "data/videos/animate_positions"
video_dirs = "data/eigd/handball/video"
vdep_scroll_dir = f"data/videos/vdep_scroll/{DATE}_predict_{PREDICT_EVENT_NUM}events"
concat_video_output_dir = f"data/videos/concat_positions_vdep/{DATE}_predict_{PREDICT_EVENT_NUM}events"

# Step 12: Create player position animations
position_animation_output_dir = "./data/videos/animate_positions"

# Step 13: Plot player velocity
velocity_plots_output_dir = "./data/fig/velocity_plots"
player_id = 0  # Player ID to process
team = "team_a"  # Team ('team_a' or 'team_b')

# Step 14: Concatenate game and velocity videos
velocity_image_path = f"./data/fig/velocity_plots/48dcd3_00-25-00.png"
velocity_scroll_video = f"data/videos/velocity_scroll_{DATE}/48dcd3_00-25-00.mp4"
original_video_path_step14 = "data/eigd/handball/video/eigd-h_vid_part_1/48dcd3_00-25-00.mp4"
final_video_output_path_step14 = f"data/videos/concat_game_velocity_{DATE}/48dcd3_00-25-00.mp4"

# Step 15: Overlay position animations on game videos
main_video_path_step15 = f"data/videos/concat_game_velocity_{DATE}/48dcd3_00-25-00.mp4"
overlay_video_path = f"data/videos/animate_positions/48dcd3_00-25-00.mp4"
overlay_output_path = f"data/videos/concat_game_velocity_positions_{DATE}/48dcd3_00-25-00.mp4"

# Step 16: Concatenate videos horizontally
main_video_path_step16 = f"data/videos/concat_positions_vdep/20250119"
concat_video_path_step16 = f"data/videos/animate_positions"
horizontal_concat_output = f"data/videos/concat_game_vdep_positions_{DATE}"

# Step 17: Plot feature importance
feature_importance_dir = f"data/csv_files/feature_importances/{DATE}_predict_{PREDICT_EVENT_NUM}events"
feature_importance_plots_output_dir = f"data/fig/feature_importances/{DATE}_predict_{PREDICT_EVENT_NUM}events"

skip_json_files = ["ad969d_00-43-00.json", 
                  "e0e547_01-00-00.json",
                  "48dcd3_00-15-00.json",
                  "labeled_features_ad969d_00-43-00.json",
                  "labeled_features_e0e547_01-00-00.json",
                  "labeled_features_48dcd3_00-15-00.json"]

skip_mp4_files = ["ad969d_00-43-00.mp4", 
                  "e0e547_01-00-00.mp4",
                  "48dcd3_00-15-00.mp4"]
# === Main Code ===

# Step 1: Convert event annotation data to SPADL format
if step1:
    print("Convert event annotation data to SPADL format")
    df = pd.read_json(json_path, orient='records', lines=True)
    spadl = cta.convert_to_spadl(df, spadl_output_dir)

# Step 2: Convert event data to features
if step2:
    print("Convert event data to features")
    spadls = os.listdir(spadl_dir)
    for spadl_file in tqdm(spadls):
        game_id = re.split('[_.]', spadl_file)[0]
        subset_id = re.split('[_.]', spadl_file)[1]
        with open(os.path.join(spadl_dir, spadl_file), 'rb') as f:
            spadl = pd.read_json(f)
        cta.spadls_to_features(spadl, features_output_dir, game_id, subset_id)

# Step 3: Label features
if step3:
    print("Label features")
    features_files = os.listdir(features_dir)
    for features_file in tqdm(features_files):
        game_id = re.split('[_.]', features_file)[0]
        subset_id = re.split('[_.]', features_file)[1]
        with open(os.path.join(features_dir, features_file), 'rb') as f:
            features = pd.read_json(f)
        cta.label_features(features, labeled_features_output_dir, game_id, subset_id, PREDICT_EVENT_NUM)

# Step 4: Feature selection
if step4:
    print("\nSTEP4: Feature Selection")
    labeled_features_files = os.listdir(labeled_features_dir)
    # データを統合
    trains = pd.DataFrame()
    for file_name in tqdm(labeled_features_files, desc="Loading data"):
        if file_name in skip_json_files:
            continue
        with open(os.path.join(labeled_features_dir, file_name), 'rb') as f:
            train = pd.read_json(f)
            trains = pd.concat([trains, train], ignore_index=True)
    # UNUSED_COLSをdrop
    trains = trains.drop(columns=UNUSED_COLS, errors='ignore')
    try:
        # Optunaによる特徴量選択の実行
        select_features_by_optuna(trains, selected_features_output_dir)
        print("Feature selection completed successfully")
        
        # 特徴量の保存先の表示
        print(f"Selected features are saved in ./data/features/{DATE}/[target_col]/optuna/selected_features.txt")
    except Exception as e:
        print(f"Error in feature selection: {e}")
        raise

# Step 5: Calculate probability
if step5:
    print("Calculate probability")
    game_ids = ['48dcd3', 'ad969d', 'e0e547', 'e8a35a', 'ec7a6a']
    labeled_features_files = os.listdir(labeled_features_dir)
    unused_cols = ["possession_team", "result", "pre_result"]
    method = 'optuna'
    param = None

    valid_files = [f for f in labeled_features_files if f not in skip_json_files]

    for test_game_id in tqdm(game_ids, desc="Processing games"):
        # 訓練データを一度に作成
        trains = pd.DataFrame()
        train_game_ids = [gid for gid in game_ids if gid != test_game_id]
        for train_game_id in train_game_ids:
            train_files = [f for f in valid_files if train_game_id in f]
            for train_file in train_files:
                train_df = pd.read_json(os.path.join(labeled_features_dir, train_file))
                trains = pd.concat([trains, train_df], ignore_index=True)

        # ここで学習を1回だけ実行
        trained_models = train_models(
            train_data=trains,
            feature_dir=selected_features_dir,
            feature_importance_output_dir=feature_importance_output_dir,
            game_id=test_game_id,
            method=method,
            param=param,
            unused_cols=unused_cols
        )

        # テストデータ(サブセット単位)で予測のみ
        test_files = [f for f in valid_files if test_game_id in f]
        for test_file in test_files:
            subset_id = re.split('[_.]', test_file)[1]
            test_subset = pd.read_json(os.path.join(labeled_features_dir, test_file))
            print(f"Test subset: {test_file}")
            predict_with_trained_models(
                test_data=test_subset,
                model_dict=trained_models,
                game_id=test_game_id,
                subset_id=subset_id,
                proba_output_dir=proba_output_dir,
                f1_scores_output_path=f1_scores_output_path
            )
            
# Step 5: Calculate probability
# all_gamesのデータを使って学習を1回だけ実行し、各試合で予測
""" if step5:
    print("Calculate probability")
    game_ids = ['48dcd3', 'ad969d', 'e0e547', 'e8a35a', 'ec7a6a']
    labeled_features_files = os.listdir(labeled_features_dir)
    unused_cols = []
    method = 'optuna'
    param = None

    valid_files = [f for f in labeled_features_files if f not in skip_json_files]

    # すべての試合を訓練データにまとめる
    all_train_df = pd.DataFrame()
    for game_id in game_ids:
        train_files = [f for f in valid_files if game_id in f]
        for train_file in train_files:
            df_tmp = pd.read_json(os.path.join(labeled_features_dir, train_file))
            all_train_df = pd.concat([all_train_df, df_tmp], ignore_index=True)

    # 全試合を用いて学習を1回だけ実行
    trained_models = train_models(
        train_data=all_train_df,
        feature_dir=selected_features_dir,
        feature_importance_output_dir=feature_importance_output_dir,
        game_id="all_games",
        method=method,
        param=param,
        unused_cols=unused_cols
    )

    # 各試合で予測
    for test_game_id in tqdm(game_ids, desc="Predicting for each game"):
        test_files = [f for f in valid_files if test_game_id in f]
        for test_file in test_files:
            subset_id = re.split('[_.]', test_file)[1]
            test_subset = pd.read_json(os.path.join(labeled_features_dir, test_file))
            print(f"Predicting: {test_file}")
            predict_with_trained_models(
                test_data=test_subset,
                model_dict=trained_models,
                game_id=test_game_id,
                subset_id=subset_id,
                proba_output_dir=proba_output_dir,
                f1_scores_output_path=f1_scores_output_path
            ) """

# Step 6: Calculate VDEP
if step6:
    print("Calculate VDEP")
    # VDEPの計算と保存
    vdep = calc_vdep(proba_path, vdep_output_dir, weights_output_path)

# Step 7: Plot F1 scores
if step7:
    df = pd.read_csv(f1_scores_path)
    plot_f1_scores(df, f1_plot_output_dir)
        
# Step 8: Plot VDEP relation
if step8:
    print("Plot VDEP relation")
    vdep_files = sorted(Path(vdep_path).glob('*.json'))
    games = {}
    current_scores = {
        'team_a': TeamScores(),
        'team_b': TeamScores()
    }
    current_game_id = None

    after_foul_concede_rate_mean = load_after_foul_concede_rate_mean(weights_path)
    
    for vdep_file in tqdm(vdep_files):
        game_id = vdep_file.stem.split('_')[0]
        subset_id = vdep_file.stem.split('_')[1]
        
        # Process previous game if we've moved to a new game
        if current_game_id and current_game_id != game_id:
            games.update(process_game_data(current_scores, current_game_id))
            current_scores = {
                'team_a': TeamScores(),
                'team_b': TeamScores()
            }
        
        # Read and process current file
        vdep_df = pd.read_json(vdep_file)
        scores = calc_scores(vdep_df, game_id, subset_id, after_foul_concede_rate_mean)
        
        # Update scores
        current_scores['team_a'].vdeps.append(scores[0])
        current_scores['team_b'].vdeps.append(scores[1])
        current_scores['team_a'].concedes.append(scores[2])
        current_scores['team_b'].concedes.append(scores[3])
        current_scores['team_a'].fouls.append(scores[4])
        current_scores['team_b'].fouls.append(scores[5])
        current_scores['team_a'].concede_probs.append(scores[8])
        current_scores['team_b'].concede_probs.append(scores[9])
        current_scores['team_a'].foul_probs.append(scores[10])
        current_scores['team_b'].foul_probs.append(scores[11])
        current_scores['team_a'].fast_break_probs.append(scores[12])
        current_scores['team_b'].fast_break_probs.append(scores[13])
        current_scores['team_a'].transformed_foul_probs.append(scores[14])
        current_scores['team_b'].transformed_foul_probs.append(scores[15])
        
        current_game_id = game_id

    # Process last game
    if current_game_id:
        games.update(process_game_data(current_scores, current_game_id))

    # Plot relations
    plot_vdep_relation(games, vdep_concede_relation_output_dir)
    plot_vdep_relation_season(games, season_data_path, vdep_concede_relation_season_output_dir)

# Step 9: Plot VDEP by event
if step9:
    print("Plot VDEP by event")
    vdep_files = sorted(Path(vdep_path).glob('*.json'))
    for vdep_file in tqdm(vdep_files):
        game_id = vdep_file.stem.split('_')[0]
        subset_id = vdep_file.stem.split('_')[1]
        vdep_df = pd.read_json(vdep_file)
        plot_vdep_by_event(vdep_df, game_id, subset_id, output_dir=vdep_event_plot_output_dir)
        plot_vdep_by_event_with_event_type(vdep_df, game_id, subset_id, output_dir=vdep_event_plot_with_event_type_output_dir)

# Step 10: Create VDEP scroll animation
if step10:
    print("Create VDEP scroll animation")
    if not os.path.exists(vdep_scroll_output_dir):
        os.makedirs(vdep_scroll_output_dir)
    for image in os.listdir(vdep_event_images_dir):
        game_id = re.split('[_.]', image)[0]
        subset_id = re.split('[_.]', image)[1]
        scroll_video = f"{vdep_scroll_output_dir}/{game_id}_{subset_id}.mp4"
        animate_vdep_scroll(os.path.join(vdep_event_images_dir, image), scroll_video, legend_image=legend_image_path)
        
""" # Step 11: Concatenate videos vertically
if step11:
    print("Concatenate videos")
    
    # video_dirs と vdep_scroll_dir が正しく設定されていることを確認
    if not os.path.exists(video_dirs):
        raise NotADirectoryError(f"Video directory does not exist: {video_dirs}")
    if not os.path.exists(vdep_scroll_dir):
        raise NotADirectoryError(f"VDEP scroll directory does not exist: {vdep_scroll_dir}")

    for video in os.listdir(video_dirs):
        video_path = os.path.join(video_dirs, video)
        if (video in skip_mp4_files) or (not video.endswith(".mp4")):
            continue
        if not os.path.isfile(video_path):
            print(f"Skipping non-file entry: {video_path}")
            continue

        game_id = re.split('[_.]', video)[0]
        subset_id = re.split('[_.]', video)[1]
        if not os.path.exists(concat_video_output_dir):
            os.makedirs(concat_video_output_dir)
        original_video = video_path
        scroll_video = os.path.join(vdep_scroll_dir, f"{game_id}_{subset_id}.mp4")
        final_video = os.path.join(concat_video_output_dir, f"{game_id}_{subset_id}.mp4")
        concat_videos_vertically([original_video, scroll_video], final_video) """
# Step 11: Concatenate videos vertically
if step11:
    print("Concatenate videos")
    
    if not os.path.exists(video_dirs):
        raise NotADirectoryError(f"Video directory does not exist: {video_dirs}")
    if not os.path.exists(vdep_scroll_dir):
        raise NotADirectoryError(f"VDEP scroll directory does not exist: {vdep_scroll_dir}")

    # 全フォルダ内のビデオを処理
    for folder in os.listdir(video_dirs):
        folder_path = os.path.join(video_dirs, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # フォルダ内の各ビデオを処理
        for video in os.listdir(folder_path):
            if not video.endswith('.mp4') or video in skip_mp4_files:
                continue
                
            video_path = os.path.join(folder_path, video)
            if not os.path.isfile(video_path):
                print(f"Skipping non-file entry: {video_path}")
                continue

            # ファイル名からgame_idとsubset_idを抽出
            game_id = video.split('_')[0]
            subset_id = video.split('_')[1].split('.')[0]
            
            if not os.path.exists(concat_video_output_dir):
                os.makedirs(concat_video_output_dir)
                
            original_video = video_path
            scroll_video = os.path.join(vdep_scroll_dir, f"{game_id}_{subset_id}.mp4")
            final_video = os.path.join(concat_video_output_dir, f"{game_id}_{subset_id}.mp4")
            
            if os.path.exists(scroll_video):
                print(f"Processing: {game_id}_{subset_id}")
                concat_videos_vertically([original_video, scroll_video], final_video)
            else:
                print(f"Skipping {video} - no matching scroll video found")

# Step 12: Create player position animations
if step12:
    print("Create player position animations")
    animate_positions(h5_dir, position_animation_output_dir)

# Step 13: Plot player velocity
if step13:
    print("Plot player velocity")
    if not os.path.exists(velocity_plots_output_dir):
        os.makedirs(velocity_plots_output_dir)

    for file_name in os.listdir(h5_dir):
        if file_name.endswith(".h5"):
            h5_file_path = os.path.join(h5_dir, file_name)
            output_file = os.path.join(velocity_plots_output_dir, f"{file_name.replace('.h5', '.png')}")

            print(f"Processing file: {h5_file_path}")
            try:
                calc_and_plot_velocity(h5_file_path, player_id, team, output_file)
                print(f"Plot saved to: {output_file}")
            except Exception as e:
                print(f"Error processing {h5_file_path}: {e}")

# Step 14: Concatenate game and velocity videos
if step14:
    print("Concatenate game and velocity videos")
    animate_vdep_scroll(image=velocity_image_path, output_file=velocity_scroll_video)
    concat_videos_vertically([original_video_path_step14, velocity_scroll_video], final_video_output_path_step14)

# Step 15: Overlay position animations on game videos
if step15:
    print("Overlay position animations on game videos")
    overlay_videos(main_video_path_step15, overlay_video_path, overlay_output_path)
        
# Step 16: Concatenate videos horizontally
if step16:
    print("Concatenate videos horizontally")
    main_videos = os.listdir(main_video_path_step16)
    concat_videos = os.listdir(concat_video_path_step16)
    
    for main_video in main_videos:
        if main_video in concat_videos:
            main_video_full_path = os.path.join(main_video_path_step15, main_video)
            concat_video_full_path = os.path.join(concat_video_path_step16, main_video)
            output_path = os.path.join(horizontal_concat_output, main_video)
            concat_videos_horizontally([main_video_full_path, concat_video_full_path], output_path)

# Step 17: Plot feature importance
if step17:
    print("Plot feature importance")
    plot_feature_importance(feature_importance_dir, feature_importance_plots_output_dir)