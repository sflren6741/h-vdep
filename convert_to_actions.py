import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from script.config import FIELD_LENGTH, FIELD_WIDTH, DATE, H5_PATH
from script.utils import save_df


def convert_to_spadl(events: pd.DataFrame, output_dir) -> pd.DataFrame:
    # 各ビデオ（game_idとsubset_id）の初期のpossession_teamを指定
    initial_possession_teams = {
        ('48dcd3', '00-06-00'): 1,
        ('48dcd3', '00-15-00'): 1,
        ('48dcd3', '00-25-00'): 0,
        ('48dcd3', '01-05-00'): 0,
        ('48dcd3', '01-10-00'): 1,
        ('ad969d', '00-00-30'): 1,
        ('ad969d', '00-15-00'): 0,
        ('ad969d', '00-43-00'): 0,
        ('ad969d', '01-11-00'): 1,
        ('ad969d', '01-35-00'): 1,
        ('e0e547', '00-00-00'): 1,
        ('e0e547', '00-08-00'): 1,
        ('e0e547', '00-15-00'): 0,
        ('e0e547', '00-50-00'): 1,
        ('e0e547', '01-00-00'): 0,
        ('e8a35a', '00-02-00'): 1,
        ('e8a35a', '00-07-00'): 0,
        ('e8a35a', '00-14-00'): 0,
        ('e8a35a', '01-05-00'): 1,
        ('e8a35a', '01-14-00'): 1,
        ('ec7a6a', '00-30-00'): 0,
        ('ec7a6a', '00-53-00'): 1,
        ('ec7a6a', '01-19-00'): 1,
        ('ec7a6a', '01-30-00'): 0,
        ('ec7a6a', '01-40-00'): 1,
    }
    spadls = pd.DataFrame(columns=["start_time", "player_id", "possession_team", "start_x", "start_y",
                                   "event_type", "result", "bodypart"])
    possession_team = None
    previous_game_id = None
    previous_subset_id = None


    for index, event in tqdm(events.iterrows()):
        # アノテーターBとCのイベントをスキップ
        if event.annotator in ('B', 'C'):
            continue
        
        game_id = event.match_id
        subset_id = event.subset_id

        # 新しいビデオの開始時にpossession_teamを初期化
        if (previous_game_id != game_id) or (previous_subset_id != subset_id):
            # 前のビデオのデータを保存
            if not spadls.empty and previous_game_id is not None and previous_subset_id is not None:
                save_df(spadls, output_dir, previous_game_id, previous_subset_id)
                spadls = pd.DataFrame(columns=["start_time", "player_id", "possession_team", "start_x", "start_y",
                                               "event_type", "result", "bodypart"])
                
            key = (game_id, subset_id)
            possession_team = initial_possession_teams.get(key, None)
            if possession_team is None:
                print(f"Warning: initial possession_team not set for {key}")
        
            previous_game_id = game_id
            previous_subset_id = subset_id
        
        start_time = int(event.t_start) if 0 <= int(event.t_start) < 9000 else 8999
        event_split = event.label_flat.split('.')
        
        if "pass" in event_split:
            event_type = "pass"
            result = "successful" if "successful" in event_split else "off target" if "off target" in event_split else "intercepted"
        elif "ball reception" in event_split:
            event_type = "reception"
            result = ""
        elif "throw-in" in event_split:
            event_type = "throw_in"
            result = ""
        elif "sanction" in event_split:
            event_type = "sanction"
            result = "two_min" if "two min" in event_split else "yellow" if "yellow" in event_split else ""
        elif "after foul" in event_split:
            event_type = "after_foul"
            result = "free_throw" if "free-throw/kick" in event_split else "penalty" if "penalty" in event_split else ""
        elif "foul" in event_split and "referee decision" in event_split:
            event_type = "foul"
            result = "successful"
        elif "shot" in event_split:
            event_type = "shot"
            result = "successful" if "successful" in event_split else "off target" if "off target" in event_split or "goal frame" in event_split else "intercepted"
        elif "kick-off" in event_split:
            event_type = "throw_off"
            result = ""
        elif "goal" in event_split:
            event_type = "goal"
            result = ""
        elif "ball out of field" in event_split:
            event_type = "ball_out"
            result = ""
        elif "ball possession" in event_split:
            event_type = "possession_change"
            possession_team = 0 if "team A" in event_split else 1
            result = ""
        else:
            event_type = "other"
            result = ""
            
        
        video_path = os.path.join(H5_PATH, f"{game_id}_{subset_id}.h5")
        with h5py.File(video_path, "r") as h5:
            # ボールの位置を取得
            if np.all(np.isnan(h5['balls'][start_time])):
                start_x, start_y = np.nan, np.nan
                player_id = np.nan
            else:
                for ball_xy_ in h5['balls'][start_time]:
                    if not np.all(np.isnan(ball_xy_)):
                        ball_xy = ball_xy_[0:2]
                        break
                dist_team_a = [np.linalg.norm(player_xy - ball_xy) for player_xy in h5['team_a'][start_time]]
                dist_team_b = [np.linalg.norm(player_xy - ball_xy) for player_xy in h5['team_b'][start_time]]

                nearest_player_index_team_a = np.nanargmin(dist_team_a)
                nearest_player_index_team_b = np.nanargmin(dist_team_b)

                if event_type == "foul":
                    if possession_team == 0:
                        player_id = int(nearest_player_index_team_b)
                        start_x = h5['team_b'][start_time][nearest_player_index_team_b][0]
                        start_y = h5['team_b'][start_time][nearest_player_index_team_b][1]
                    else:
                        player_id = int(nearest_player_index_team_a)
                        start_x = h5['team_a'][start_time][nearest_player_index_team_a][0]
                        start_y = h5['team_a'][start_time][nearest_player_index_team_a][1]
                else:
                    if possession_team == 0:
                        player_id = int(nearest_player_index_team_a)
                        start_x = h5['team_a'][start_time][nearest_player_index_team_a][0]
                        start_y = h5['team_a'][start_time][nearest_player_index_team_a][1]
                    else:
                        player_id = int(nearest_player_index_team_b)
                        start_x = h5['team_b'][start_time][nearest_player_index_team_b][0]
                        start_y = h5['team_b'][start_time][nearest_player_index_team_b][1]
                        
        bodypart = "hand"
        spadl = pd.DataFrame([[start_time, player_id, possession_team, start_x, start_y, event_type, result, bodypart]], 
                              columns=["start_time", "player_id", "possession_team", "start_x", "start_y", 
                                       "event_type", "result", "bodypart"])
        spadls = pd.concat([spadls, spadl], ignore_index=True)

    # 最後のビデオのデータを保存
    if not spadls.empty and previous_game_id is not None and previous_subset_id is not None:
        save_df(spadls, output_dir, previous_game_id, previous_subset_id)
    
def spadls_to_features(spadls: pd.DataFrame, output_dir, game_id, subset_id) -> pd.DataFrame:
    # 修正された列名の順序
    label_cols = [
        "event_type", "result", "possession_team",
        "start_time", "ball_x", "ball_y", "distance_to_goal", "goal_angle",
        "pre_start_time", "pre_ball_x", "pre_ball_y", "pre_distance_to_goal", "pre_goal_angle",
        "possession_duration"
    ]

    sorted_cols = [
        f"ball_2_atk_{i}" for i in range(7)
    ] + [
        f"ball_2_dfd_{i}" for i in range(7)
    ] + [
        f"atk_{i}_x" for i in range(7)
    ] + [
        f"atk_{i}_y" for i in range(7)
    ] + [
        f"dfd_{i}_x" for i in range(7)
    ] + [
        f"dfd_{i}_y" for i in range(7)
    ] + [
        f"atk_{i}_vx" for i in range(7)
    ] + [
        f"atk_{i}_vy" for i in range(7)
    ] + [
        f"dfd_{i}_vx" for i in range(7)
    ] + [
        f"dfd_{i}_vy" for i in range(7)
    ]

    features = pd.DataFrame(columns=label_cols + sorted_cols)
    
    # 最初のポゼッションチームとその時間を記録
    last_possession_team = spadls.iloc[0]['possession_team']
    last_possession_change_time = spadls.iloc[0]['start_time']
    
    video_path = os.path.join(H5_PATH, f"{game_id}_{subset_id}.h5")
    with h5py.File(video_path, "r") as h5:
        for i, spadl in spadls.iterrows():
            event_type = spadl['event_type']
            result = spadl['result']
            possession_team = spadl['possession_team']
            start_time = spadl['start_time']
            # ボールの位置を取得
            ball_positions = h5['balls'][start_time][:, :2]

            # 有効なボールのインデックスを取得
            valid_indices = np.where(np.isfinite(ball_positions).all(axis=1))[0]
            ball_pos = ball_positions[valid_indices[0]] if len(valid_indices) > 0 else np.array([np.nan, np.nan])
            
            # ポゼッションの変更をチェック
            if possession_team != last_possession_team:
                last_possession_change_time = start_time
                last_possession_team = possession_team
            elapsed_time_since_possession_change = start_time - last_possession_change_time
            
            # 座標の反転条件
            flip_condition = (
                (game_id == "48dcd3" and subset_id in ["00-06-00", "00-15-00", "00-25-00"]) or
                (game_id == "ad969d" and subset_id in ["01-11-00", "01-35-00"]) or
                (game_id == "e0e547" and subset_id in ["00-00-00", "00-08-00", "00-15-00"]) or
                (game_id == "e8a35a" and subset_id in ["00-02-00", "00-07-00", "00-14-00"]) or
                (game_id == "ec7a6a" and subset_id in ["00-30-00", "00-53-00"])
            )
            
            # ボールの位置をポゼッションチームとゲーム条件に基づいて処理
            if flip_condition:
                if possession_team == 0:
                    ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - ball_pos
            else:
                if possession_team != 0:
                    ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - ball_pos

            # ゴールまでの距離と角度を計算
            goal_pos = np.array([FIELD_LENGTH, int(FIELD_WIDTH / 2)])
            distance_to_goal = np.linalg.norm(ball_pos - goal_pos)
            goal_direction = np.array([0, -1])  # ゴールは下方向にあると仮定
            ball_to_goal = goal_pos - ball_pos
            goal_angle = np.arccos(np.clip(
                np.dot(ball_to_goal, goal_direction) /
                (np.linalg.norm(ball_to_goal) * np.linalg.norm(goal_direction)), -1.0, 1.0
            ))
            
            curr_time_related = [start_time] + ball_pos.tolist() + [distance_to_goal, goal_angle]

            # 前のボール位置関連の情報を計算
            if i == 0:
                pre_time_related = [np.nan] * 5
            else:
                prev_spadl = spadls.iloc[i-1]
                pre_start_time = prev_spadl['start_time']
                pre_possession_team = prev_spadl['possession_team']
                pre_ball_positions = h5['balls'][pre_start_time][:, :2]
                valid_indices = np.where(np.isfinite(pre_ball_positions).all(axis=1))[0]
                pre_ball_pos = pre_ball_positions[valid_indices[0]] if len(valid_indices) > 0 else np.array([np.nan, np.nan])
                    
                # ボール位置の反転処理
                if flip_condition:
                    if pre_possession_team == 0:
                        pre_ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - pre_ball_pos
                else:
                    if pre_possession_team != 0:
                        pre_ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - pre_ball_pos

                pre_distance_to_goal = np.linalg.norm(pre_ball_pos - goal_pos)
                pre_ball_to_goal = goal_pos - pre_ball_pos
                pre_goal_angle = np.arccos(np.clip(
                    np.dot(pre_ball_to_goal, goal_direction) /
                    (np.linalg.norm(pre_ball_to_goal) * np.linalg.norm(goal_direction)), -1.0, 1.0
                ))
                pre_time_related = [pre_start_time] + pre_ball_pos.tolist() + [pre_distance_to_goal, pre_goal_angle]

            # 選手の位置、速度、距離の計算
            if start_time >= len(h5['team_a']) or start_time >= len(h5['team_b']):
                continue  # データが存在しない場合はスキップ

            atk_team = 'team_a' if possession_team == 0 else 'team_b'
            dfd_team = 'team_b' if possession_team == 0 else 'team_a'

            atk_positions = h5[atk_team][start_time]
            dfd_positions = h5[dfd_team][start_time]

            # 座標の反転処理
            if flip_condition:
                if possession_team == 0:
                    atk_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_positions
                    dfd_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_positions
            else:
                if possession_team != 0:
                    atk_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_positions
                    dfd_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_positions

            # 前フレームの位置取得（速度計算用）
            pre_time_candidate = start_time - 1
            if pre_time_candidate >= 0:
                atk_pre_positions = h5[atk_team][pre_time_candidate]
                dfd_pre_positions = h5[dfd_team][pre_time_candidate]

                # 座標の反転処理
                if flip_condition:
                    if possession_team == 0:
                        atk_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_pre_positions
                        dfd_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_pre_positions
                else:
                    if possession_team != 0:
                        atk_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_pre_positions
                        dfd_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_pre_positions
            else:
                atk_pre_positions = np.full_like(atk_positions, np.nan)
                dfd_pre_positions = np.full_like(dfd_positions, np.nan)

            # 攻撃側の選手の位置、速度、距離を計算
            atk_distances = []
            atk_positions_list = []
            atk_velocities = []
            for pos, pre_pos in zip(atk_positions, atk_pre_positions):
                if not np.any(np.isnan(pos)):
                    distance = np.linalg.norm(pos - ball_pos)
                    atk_distances.append(distance)
                    atk_positions_list.append(pos.tolist())

                    if not np.any(np.isnan(pre_pos)):
                        delta_pos = pos - pre_pos
                        delta_time = start_time - pre_time_candidate
                        velocity = delta_pos / delta_time if delta_time > 0 else np.array([np.nan, np.nan])
                    else:
                        velocity = np.array([np.nan, np.nan])
                    atk_velocities.append(velocity.tolist())
                else:
                    atk_distances.append(np.nan)
                    atk_positions_list.append([np.nan, np.nan])
                    atk_velocities.append([np.nan, np.nan])

            # 守備側の選手の位置、速度、距離を計算
            dfd_distances = []
            dfd_positions_list = []
            dfd_velocities = []
            for pos, pre_pos in zip(dfd_positions, dfd_pre_positions):
                if not np.any(np.isnan(pos)):
                    distance = np.linalg.norm(pos - ball_pos)
                    dfd_distances.append(distance)
                    dfd_positions_list.append(pos.tolist())

                    if not np.any(np.isnan(pre_pos)):
                        delta_pos = pos - pre_pos
                        delta_time = start_time - pre_time_candidate
                        velocity = delta_pos / delta_time if delta_time > 0 else np.array([np.nan, np.nan])
                    else:
                        velocity = np.array([np.nan, np.nan])
                    dfd_velocities.append(velocity.tolist())
                else:
                    dfd_distances.append(np.nan)
                    dfd_positions_list.append([np.nan, np.nan])
                    dfd_velocities.append([np.nan, np.nan])

            # ボールからの距離でソートし、上位7人のデータを取得
            atk_sorted_indices = np.argsort(atk_distances)[:7]
            dfd_sorted_indices = np.argsort(dfd_distances)[:7]

            sorted_ball2_atk = [atk_distances[i] if not np.isnan(atk_distances[i]) else np.nan for i in atk_sorted_indices]
            sorted_ball2_dfd = [dfd_distances[i] if not np.isnan(dfd_distances[i]) else np.nan for i in dfd_sorted_indices]

            atk_nearest_positions = [atk_positions_list[i] for i in atk_sorted_indices]
            dfd_nearest_positions = [dfd_positions_list[i] for i in dfd_sorted_indices]

            atk_nearest_velocities = [atk_velocities[i] for i in atk_sorted_indices]
            dfd_nearest_velocities = [dfd_velocities[i] for i in dfd_sorted_indices]

            # リストの長さを7に調整
            while len(sorted_ball2_atk) < 7:
                sorted_ball2_atk.append(np.nan)
                atk_nearest_positions.append([np.nan, np.nan])
                atk_nearest_velocities.append([np.nan, np.nan])

            while len(sorted_ball2_dfd) < 7:
                sorted_ball2_dfd.append(np.nan)
                dfd_nearest_positions.append([np.nan, np.nan])
                dfd_nearest_velocities.append([np.nan, np.nan])

            # データフレームにデータを追加
            feature_data = (
                [event_type, result, possession_team] +
                curr_time_related +
                pre_time_related +
                [elapsed_time_since_possession_change] +
                sorted_ball2_atk +
                sorted_ball2_dfd +
                [atk_nearest_positions[i][0] for i in range(7)] +  # atk_nearest_x
                [atk_nearest_positions[i][1] for i in range(7)] +  # atk_nearest_y
                [dfd_nearest_positions[i][0] for i in range(7)] +  # dfd_nearest_x
                [dfd_nearest_positions[i][1] for i in range(7)] +  # dfd_nearest_y
                [atk_nearest_velocities[i][0] for i in range(7)] +  # atk_nearest_vx
                [atk_nearest_velocities[i][1] for i in range(7)] +  # atk_nearest_vy
                [dfd_nearest_velocities[i][0] for i in range(7)] +  # dfd_nearest_vx
                [dfd_nearest_velocities[i][1] for i in range(7)]  # dfd_nearest_vy
            )

            feature = pd.DataFrame([feature_data], columns=label_cols + sorted_cols)
            features = pd.concat([features, feature], ignore_index=True)
    
    save_df(features, output_dir, game_id, subset_id)

""" def spadls_to_features(spadls: pd.DataFrame, output_dir, game_id, subset_id) -> pd.DataFrame:
    # 修正された列名の順序
    label_cols = [
        "event_type", "result", "possession_team",
        "start_time", "ball_x", "ball_y", "distance_to_goal", "goal_angle",
        "pre_start_time", "pre_event_type", "pre_result", "pre_ball_x", "pre_ball_y", "pre_distance_to_goal", "pre_goal_angle",
        "possession_duration"
    ]

    sorted_cols = [
        f"ball_2_atk_{i}" for i in range(7)
    ] + [
        f"ball_2_dfd_{i}" for i in range(7)
    ] + [
        f"atk_{i}_x" for i in range(7)
    ] + [
        f"atk_{i}_y" for i in range(7)
    ] + [
        f"dfd_{i}_x" for i in range(7)
    ] + [
        f"dfd_{i}_y" for i in range(7)
    ] + [
        f"atk_{i}_vx" for i in range(7)
    ] + [
        f"atk_{i}_vy" for i in range(7)
    ] + [
        f"dfd_{i}_vx" for i in range(7)
    ] + [
        f"dfd_{i}_vy" for i in range(7)
    ]

    features = pd.DataFrame(columns=label_cols + sorted_cols)
    
    # 最初のポゼッションチームとその時間を記録
    last_possession_team = spadls.iloc[0]['possession_team']
    last_possession_change_time = spadls.iloc[0]['start_time']
    
    video_path = os.path.join(H5_PATH, f"{game_id}_{subset_id}.h5")
    with h5py.File(video_path, "r") as h5:
        for i, spadl in spadls.iterrows():
            event_type = spadl['event_type']
            result = spadl['result']
            possession_team = spadl['possession_team']
            start_time = spadl['start_time']
            # ボールの位置を取得
            ball_positions = h5['balls'][start_time][:, :2]

            # 有効なボールのインデックスを取得
            valid_indices = np.where(np.isfinite(ball_positions).all(axis=1))[0]
            ball_pos = ball_positions[valid_indices[0]] if len(valid_indices) > 0 else np.array([np.nan, np.nan])
            
            # ポゼッションの変更をチェック
            if possession_team != last_possession_team:
                last_possession_change_time = start_time
                last_possession_team = possession_team
            elapsed_time_since_possession_change = start_time - last_possession_change_time
            
            # 座標の反転条件
            flip_condition = (
                (game_id == "48dcd3" and subset_id in ["00-06-00", "00-15-00", "00-25-00"]) or
                (game_id == "ad969d" and subset_id in ["01-11-00", "01-35-00"]) or
                (game_id == "e0e547" and subset_id in ["00-00-00", "00-08-00", "00-15-00"]) or
                (game_id == "e8a35a" and subset_id in ["00-02-00", "00-07-00", "00-14-00"]) or
                (game_id == "ec7a6a" and subset_id in ["00-30-00", "00-53-00"])
            )
            
            # ボールの位置をポゼッションチームとゲーム条件に基づいて処理
            if flip_condition:
                if possession_team == 0:
                    ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - ball_pos
            else:
                if possession_team != 0:
                    ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - ball_pos

            # ゴールまでの距離と角度を計算
            goal_pos = np.array([FIELD_LENGTH, int(FIELD_WIDTH / 2)])
            distance_to_goal = np.linalg.norm(ball_pos - goal_pos)
            goal_direction = np.array([0, -1])  # ゴールは下方向にあると仮定
            ball_to_goal = goal_pos - ball_pos
            goal_angle = np.arccos(np.clip(
                np.dot(ball_to_goal, goal_direction) /
                (np.linalg.norm(ball_to_goal) * np.linalg.norm(goal_direction)), -1.0, 1.0
            ))
            
            curr_time_related = [start_time] + ball_pos.tolist() + [distance_to_goal, goal_angle]

            # 前のボール位置関連の情報を計算
            if i == 0:
                pre_time_related = [np.nan] * 7
            else:
                prev_spadl = spadls.iloc[i-1]
                pre_start_time = prev_spadl['start_time']
                pre_event_type = prev_spadl['event_type']
                pre_result = prev_spadl['result']
                pre_possession_team = prev_spadl['possession_team']
                pre_ball_positions = h5['balls'][pre_start_time][:, :2]
                valid_indices = np.where(np.isfinite(pre_ball_positions).all(axis=1))[0]
                pre_ball_pos = pre_ball_positions[valid_indices[0]] if len(valid_indices) > 0 else np.array([np.nan, np.nan])
                    
                # ボール位置の反転処理
                if flip_condition:
                    if pre_possession_team == 0:
                        pre_ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - pre_ball_pos
                else:
                    if pre_possession_team != 0:
                        pre_ball_pos = np.array([FIELD_LENGTH, FIELD_WIDTH]) - pre_ball_pos

                pre_distance_to_goal = np.linalg.norm(pre_ball_pos - goal_pos)
                pre_ball_to_goal = goal_pos - pre_ball_pos
                pre_goal_angle = np.arccos(np.clip(
                    np.dot(pre_ball_to_goal, goal_direction) /
                    (np.linalg.norm(pre_ball_to_goal) * np.linalg.norm(goal_direction)), -1.0, 1.0
                ))
                pre_time_related = [pre_start_time, pre_event_type, pre_result] + pre_ball_pos.tolist() + [pre_distance_to_goal, pre_goal_angle]

            # 選手の位置、速度、距離の計算
            if start_time >= len(h5['team_a']) or start_time >= len(h5['team_b']):
                continue  # データが存在しない場合はスキップ

            atk_team = 'team_a' if possession_team == 0 else 'team_b'
            dfd_team = 'team_b' if possession_team == 0 else 'team_a'

            atk_positions = h5[atk_team][start_time]
            dfd_positions = h5[dfd_team][start_time]

            # 座標の反転処理
            if flip_condition:
                if possession_team == 0:
                    atk_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_positions
                    dfd_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_positions
            else:
                if possession_team != 0:
                    atk_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_positions
                    dfd_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_positions

            # 前フレームの位置取得（速度計算用）
            pre_time_candidate = start_time - 1
            if pre_time_candidate >= 0:
                atk_pre_positions = h5[atk_team][pre_time_candidate]
                dfd_pre_positions = h5[dfd_team][pre_time_candidate]

                # 座標の反転処理
                if flip_condition:
                    if possession_team == 0:
                        atk_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_pre_positions
                        dfd_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_pre_positions
                else:
                    if possession_team != 0:
                        atk_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - atk_pre_positions
                        dfd_pre_positions = np.array([FIELD_LENGTH, FIELD_WIDTH]) - dfd_pre_positions
            else:
                atk_pre_positions = np.full_like(atk_positions, np.nan)
                dfd_pre_positions = np.full_like(dfd_positions, np.nan)

            # 攻撃側の選手の位置、速度、距離を計算
            atk_distances = []
            atk_positions_list = []
            atk_velocities = []
            for pos, pre_pos in zip(atk_positions, atk_pre_positions):
                if not np.any(np.isnan(pos)):
                    distance = np.linalg.norm(pos - ball_pos)
                    atk_distances.append(distance)
                    atk_positions_list.append(pos.tolist())

                    if not np.any(np.isnan(pre_pos)):
                        delta_pos = pos - pre_pos
                        delta_time = start_time - pre_time_candidate
                        velocity = delta_pos / delta_time if delta_time > 0 else np.array([np.nan, np.nan])
                    else:
                        velocity = np.array([np.nan, np.nan])
                    atk_velocities.append(velocity.tolist())
                else:
                    atk_distances.append(np.nan)
                    atk_positions_list.append([np.nan, np.nan])
                    atk_velocities.append([np.nan, np.nan])

            # 守備側の選手の位置、速度、距離を計算
            dfd_distances = []
            dfd_positions_list = []
            dfd_velocities = []
            for pos, pre_pos in zip(dfd_positions, dfd_pre_positions):
                if not np.any(np.isnan(pos)):
                    distance = np.linalg.norm(pos - ball_pos)
                    dfd_distances.append(distance)
                    dfd_positions_list.append(pos.tolist())

                    if not np.any(np.isnan(pre_pos)):
                        delta_pos = pos - pre_pos
                        delta_time = start_time - pre_time_candidate
                        velocity = delta_pos / delta_time if delta_time > 0 else np.array([np.nan, np.nan])
                    else:
                        velocity = np.array([np.nan, np.nan])
                    dfd_velocities.append(velocity.tolist())
                else:
                    dfd_distances.append(np.nan)
                    dfd_positions_list.append([np.nan, np.nan])
                    dfd_velocities.append([np.nan, np.nan])

            # ボールからの距離でソートし、上位7人のデータを取得
            atk_sorted_indices = np.argsort(atk_distances)[:7]
            dfd_sorted_indices = np.argsort(dfd_distances)[:7]

            sorted_ball2_atk = [atk_distances[i] if not np.isnan(atk_distances[i]) else np.nan for i in atk_sorted_indices]
            sorted_ball2_dfd = [dfd_distances[i] if not np.isnan(dfd_distances[i]) else np.nan for i in dfd_sorted_indices]

            atk_nearest_positions = [atk_positions_list[i] for i in atk_sorted_indices]
            dfd_nearest_positions = [dfd_positions_list[i] for i in dfd_sorted_indices]

            atk_nearest_velocities = [atk_velocities[i] for i in atk_sorted_indices]
            dfd_nearest_velocities = [dfd_velocities[i] for i in dfd_sorted_indices]

            # リストの長さを7に調整
            while len(sorted_ball2_atk) < 7:
                sorted_ball2_atk.append(np.nan)
                atk_nearest_positions.append([np.nan, np.nan])
                atk_nearest_velocities.append([np.nan, np.nan])

            while len(sorted_ball2_dfd) < 7:
                sorted_ball2_dfd.append(np.nan)
                dfd_nearest_positions.append([np.nan, np.nan])
                dfd_nearest_velocities.append([np.nan, np.nan])

            # データフレームにデータを追加
            feature_data = (
                [event_type, result, possession_team] +
                curr_time_related +
                pre_time_related +
                [elapsed_time_since_possession_change] +
                sorted_ball2_atk +
                sorted_ball2_dfd +
                [atk_nearest_positions[i][0] for i in range(7)] +  # atk_nearest_x
                [atk_nearest_positions[i][1] for i in range(7)] +  # atk_nearest_y
                [dfd_nearest_positions[i][0] for i in range(7)] +  # dfd_nearest_x
                [dfd_nearest_positions[i][1] for i in range(7)] +  # dfd_nearest_y
                [atk_nearest_velocities[i][0] for i in range(7)] +  # atk_nearest_vx
                [atk_nearest_velocities[i][1] for i in range(7)] +  # atk_nearest_vy
                [dfd_nearest_velocities[i][0] for i in range(7)] +  # dfd_nearest_vx
                [dfd_nearest_velocities[i][1] for i in range(7)]  # dfd_nearest_vy
            )

            feature = pd.DataFrame([feature_data], columns=label_cols + sorted_cols)
            features = pd.concat([features, feature], ignore_index=True)
    
    save_df(features, output_dir, game_id, subset_id) """
    
def label_features(features: pd.DataFrame, output_dir, game_id, subset_id, predict_event_num) -> pd.DataFrame:
    features['concede_future_label'] = 0
    features['foul_future_label'] = 0
    features['fast_break_future_label'] = 0

    # 速攻のラベル付け
    possession_start_time = None
    possession_change_idx = None
    previous_possession_change_idx = None
    for i in range(features.shape[0]):
        if features.at[i, 'event_type'] == 'possession_change':
            possession_start_time = (features.at[i, 'start_time'] - 1) / 30
            if possession_change_idx is not None:
                previous_possession_change_idx = possession_change_idx
            possession_change_idx = i
            
        elif features.at[i, 'event_type'] == 'shot' and possession_start_time is not None:
            shot_time = (features.at[i, 'start_time'] - 1) / 30
            if (shot_time - possession_start_time) < 15.0:
                # 速攻が発生した場合、その前のポゼッションチェンジに対してラベルを付ける
                j = i
                while j > 0 and features.at[j, 'event_type'] != 'possession_change':
                    j -= 1
                if features.at[j, 'event_type'] == 'possession_change':
                    start_idx = max(0, j - predict_event_num)
                    if previous_possession_change_idx is None or start_idx > previous_possession_change_idx:
                        features.loc[start_idx:j-1, 'fast_break_future_label'] = 1
                    else:
                        features.loc[previous_possession_change_idx:j-1, 'fast_break_future_label'] = 1

    # ファウルのラベル付け
    foul_event_num = 3
    to_drop = []
    for i in range(features.shape[0]):
        if features.at[i, 'event_type'] == 'foul':
            converted = False  # フリースローへの変換が行われたかどうかを追跡
            has_penalty = False
            has_possession_change = False
            has_free_throw = False
            for j in range(i + 1, min(i + 1 + foul_event_num, features.shape[0])):
                if features.at[j, 'event_type'] == 'after_foul':
                    if features.at[j, 'result'] == 'penalty':
                        has_penalty = True
                    elif features.at[j, 'result'] == 'free_throw':
                        has_free_throw = True
                    to_drop.append(j)
                elif features.at[j, 'event_type'] == 'possession_change':
                    has_possession_change = True
                elif features.at[j, 'event_type'] == 'sanction':
                    if features.at[j, 'result'] == 'two_min':
                        features.at[i, 'result'] = 'two_min'
                    elif features.at[j, 'result'] == 'yellow':
                        features.at[i, 'result'] = 'yellow'
                    to_drop.append(j)
                elif features.at[j, 'event_type'] == 'pass' and not converted:
                    features.at[j, 'event_type'] = 'free_throw'
                    converted = True  # 一度変換したらフラグを立てる

            # 優先順位に基づいてイベントタイプを決定
            if has_penalty:
                features.at[i, 'event_type'] = 'foul_penalty'
            elif has_possession_change:
                features.at[i, 'event_type'] = 'foul_possession_change'
            elif has_free_throw:
                features.at[i, 'event_type'] = 'foul_free_throw'
                
    features.drop(index=to_drop, inplace=True)
    features.reset_index(drop=True, inplace=True)

    # 失点とファウルのラベル付け
    for i in range(features.shape[0]):
        for j in range(i + 1, min(i + 1 + predict_event_num, features.shape[0])):
            if features.at[i, 'possession_team'] == features.at[j, 'possession_team']:
                if features.at[j, 'event_type'] == 'goal':
                    features.at[i, 'concede_future_label'] = 1
                if features.at[j, 'event_type'] == 'foul_free_throw':
                    features.at[i, 'foul_future_label'] = 1
            else:
                break

    # pre_event_type, pre_result の追加
    features['pre_event_type'] = features['event_type'].shift(1)
    features['pre_result'] = features['result'].shift(1)

    features = features.iloc[:-predict_event_num]  # Drop last rows
    features.reset_index(drop=True, inplace=True)
    
    save_df(features, output_dir, game_id, subset_id)