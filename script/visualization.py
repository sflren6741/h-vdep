import re
import h5py
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import seaborn as sns
import cv2
from moviepy.editor import VideoClip, VideoFileClip, clips_array
from matplotlib import patches as mpatches
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from adjustText import adjust_text

from script.config import DATE, LABEL_COLS, COLORS, COLORS2

def plot_position():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
    a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)
    a3 = patches.Arc(xy=(40.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=90.0, theta2=180.0)
    a4 = patches.Arc(xy=(40.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=180.0, theta2=270.0)
    h5_path = "./data/eigd/handball/positions/"
    positions = os.listdir(h5_path)
    
    for position in positions:
        print(position)
        game_id = re.split('[_.]', position)[0]
        subset_id = re.split('[_.]', position)[1]
        save_dir = f"./data/plot_positions/{game_id}_{subset_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(0.0, 20.0)
        ax.add_patch(a1)
        ax.add_patch(a2)
        ax.add_patch(a3)
        ax.add_patch(a4)
        ax.plot([6.0, 6.0], [8.5, 11.5], color="k")
        ax.plot([34.0, 34.0], [8.5, 11.5], color="k")
        ax.plot([20.0, 20.0], [0.0, 20.0], color="k")
        ax.scatter([], [], c="y", label="Ball")
        ax.scatter([], [], c="k", label="Team A")
        ax.scatter([], [], c="r", label="Team B")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        with h5py.File(os.path.join(h5_path, position), 'r') as h5:
            for i in range(0, len(h5['balls']), 90):
                ax.set_title(f"{game_id}_{subset_id}_{i}")
                im = []
                texts = []  # テキストのリストを初期化
                
                offset = 0.3  # 番号の表示をずらす距離
                
                for ball_xy in h5['balls'][i]:
                    if not np.all(np.isnan(ball_xy)):
                        im.append(ax.plot(ball_xy[0], ball_xy[1], marker='o', color='y', label="Ball"))

                for idx, team_a_xy in enumerate(h5['team_a'][i]):
                    if not np.all(np.isnan(team_a_xy)):
                        im.append(ax.plot(team_a_xy[0], team_a_xy[1], marker='o', color='k', label="Team A"))
                        texts.append(ax.text(team_a_xy[0] + offset, team_a_xy[1] + offset, str(idx), fontsize=8, color='k'))

                for idx, team_b_xy in enumerate(h5['team_b'][i]):
                    if not np.all(np.isnan(team_b_xy)):
                        im.append(ax.plot(team_b_xy[0], team_b_xy[1], marker='o', color='r', label="Team B"))
                        texts.append(ax.text(team_b_xy[0] + offset, team_b_xy[1] + offset, str(idx), fontsize=8, color='r'))

                plt.savefig(os.path.join(save_dir, f"{game_id}_{subset_id}_{i}"))
                
                # プロットを削除
                for image in im:
                    point = image.pop(0)
                    point.remove()
                
                # テキストを削除
                for text in texts:
                    text.remove()


# 特定のフレームの位置データをプロット
def plot_position_for_frame():
    fig, ax = plt.subplots(figsize=(12,4))
    # r1 = patches.Rectangle(xy=(6.0, 7.0), width=10.0, height=6.0, ec='#000000', fill=False)
    a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
    a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)
    a3 = patches.Arc(xy=(40.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=90.0, theta2=180.0)
    a4 = patches.Arc(xy=(40.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=180.0, theta2=270.0)
    h5_path = "./data/eigd/handball/positions/"
    position = "48dcd3_00-06-00.h5"
    
    game_id = re.split('[_.]', position)[0]
    subset_id = re.split('[_.]', position)[1]
    save_dir = f"./data/positions_202407301633/"
    if not os.path.exists(save_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(save_dir)
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim(-1.0,41.0)
    ax.set_ylim(-1.0,21.0)
    # ax.add_patch(r1)
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.add_patch(a3)
    ax.add_patch(a4)
    ax.plot([6.0, 6.0],[8.5, 11.5],color="k")
    ax.plot([34.0, 34.0],[8.5, 11.5],color="k")
    ax.plot([20.0, 20.0],[0.0, 20.0],color="k")
    ax.plot([0.0, 40.0],[0.0, 0.0],color="k")
    ax.plot([0.0, 40.0],[20.0, 20.0],color="k")
    ax.plot([0.0, 0.0],[0.0, 20.0],color="k")
    ax.plot([40.0, 40.0],[0.0, 20.0],color="k")
    ax.scatter([], [], c="y", label="Ball")
    ax.scatter([], [], c="k", label="Team A")
    ax.scatter([], [], c="r", label="Team B")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    start_times = [5668, 5680, 5758, 5767, 5793, 5810, 5835, 5852, 5878]
    # start_times = [5668, 5758, 5878]
    #actions = ["free_throw", "reception", "pass", "reception", "pass", "reception", "pass", "reception", "shot"]
    with h5py.File(os.path.join(h5_path, position), 'r') as h5:
        for i in start_times:
            im = []
            for ball_xy in h5['balls'][i]:
                #print(ball_xy)
                if not np.all(np.isnan(ball_xy)):
                    im.append(ax.plot(ball_xy[0], ball_xy[1], marker='o', color='y', label="Ball", markersize=8))
            for team_a_xy in h5['team_a'][i]:
                if not np.all(np.isnan(team_a_xy)):
                    im.append(ax.plot(team_a_xy[0], team_a_xy[1], marker='o', color='k', label="Team A", markersize=8))
            for team_b_xy in h5['team_b'][i]:
                if not np.all(np.isnan(team_b_xy)):
                    im.append(ax.plot(team_b_xy[0], team_b_xy[1], marker='o', color='r', label="Team B", markersize=8))
            fig.savefig(os.path.join(save_dir, f"{game_id}_{subset_id}_{i}"), dpi=300)
            for image in im:
                point = image.pop(0)
                point.remove()
                
def plot_feature_importance(feature_importance_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    targets = ['concede', 'foul', 'fast_break']

    for target in targets:
        target_files = [f for f in os.listdir(feature_importance_dir) if target in f and f.endswith('.csv')]
        if not target_files:
            print(f"Feature importance CSV for {target} not found.")
            continue
        
        # 全ての動画のデータを統合
        all_feature_importances = pd.DataFrame()
        for file in target_files:
            df = pd.read_csv(os.path.join(feature_importance_dir, file))
            all_feature_importances = pd.concat([all_feature_importances, df])
        
        # 特徴量重要度を計算
        feature_importance_summary = all_feature_importances.groupby('Feature')['Importance'].mean().reset_index()
        feature_importance_summary = feature_importance_summary.sort_values(by='Importance', ascending=False)
        
        # 上位10個の特徴量をプロット
        top_features = feature_importance_summary
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color=COLORS[0])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 10 Feature Importances for {target}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_file = os.path.join(save_dir, f"{target}_feature_importance_plot.png")
        plt.savefig(plot_file)
        plt.close()
        plt.close()

def plot_vdep_by_event(vdep, game_id, subset_id, output_dir):
    # print(f"{game_id}_{subset_id}")
    possession_team = []
    vdep_team_a = []
    vdep_team_b = []
    preds_proba_concede_team_a = []
    preds_proba_concede_team_b = []
    preds_proba_foul_team_a = []
    preds_proba_foul_team_b = []
    preds_proba_fast_break_team_a = []
    preds_proba_fast_break_team_b = []
    time_team_a = []
    time_team_b = []
    texts_team_a = []
    texts_team_b = []
    
    # plot vdep by events
    for index, row in vdep.iterrows():
        possession_team.append(row['possession_team'])
        if row['possession_team'] == 0:
            if row['event_type'] == 'possession_change':
                vdep_team_b.append(np.nan)
                preds_proba_concede_team_b.append(np.nan)
                preds_proba_foul_team_b.append(np.nan)
                preds_proba_fast_break_team_b.append(np.nan)
                time_team_b.append(row['start_time'])
                texts_team_b.append("")
            vdep_team_b.append(row['vdep'])
            preds_proba_concede_team_b.append(row['preds_proba_concede'])
            preds_proba_foul_team_b.append(row['preds_proba_foul'])
            preds_proba_fast_break_team_b.append(row['preds_proba_fast_break'])
            time_team_b.append(row['start_time'])
            texts_team_b.append(row['event_type'])
            
        else:
            if row['event_type'] == 'possession_change':
                vdep_team_a.append(np.nan)
                preds_proba_concede_team_a.append(np.nan)
                preds_proba_foul_team_a.append(np.nan)
                preds_proba_fast_break_team_a.append(np.nan)
                time_team_a.append(row['start_time'])
                texts_team_a.append("")
            vdep_team_a.append(row['vdep'])
            preds_proba_concede_team_a.append(row['preds_proba_concede'])
            preds_proba_foul_team_a.append(row['preds_proba_foul'])
            preds_proba_fast_break_team_a.append(row['preds_proba_fast_break'])
            time_team_a.append(row['start_time'])
            texts_team_a.append(row['event_type'])

    # 目標のピクセルサイズとdpi
    width_px = 9600
    height_px = 720
    dpi=100
    # 図のサイズをインチで計算
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize)
    # 余白の調整
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    alpha = 0.8
    ax2 = ax.twinx()
    ax2.plot(time_team_a, preds_proba_concede_team_a, label="Concede", color=COLORS2[2], alpha=alpha)
    ax2.plot(time_team_a, preds_proba_foul_team_a, label="Foul", color=COLORS2[3], alpha=alpha)
    ax2.plot(time_team_a, preds_proba_fast_break_team_a, label="Fast break", color=COLORS2[4], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_concede_team_b, label="Concede", color=COLORS2[2], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_foul_team_b, label="Foul", color=COLORS2[3], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_fast_break_team_b, label="Fast break", color=COLORS2[4], alpha=alpha)
    # ax2.set_ylabel('Event probability')
    # ax2.yaxis.set_visible(False)
    ax.plot(time_team_a, vdep_team_a, color=COLORS2[0], marker='o', lw=4)
    ax.plot(time_team_b, vdep_team_b, color=COLORS2[1], marker='o', lw=4)
    
    ax.plot([], [], label="Team A's VDEP", color=COLORS2[0])
    ax.plot([], [], label="Team B's VDEP", color=COLORS2[1])
    ax.plot([], [], label="Concede", color=COLORS2[2])
    ax.plot([], [], label="Foul", color=COLORS2[3])
    ax.plot([], [], label="Fast break", color=COLORS2[4])
    ax.set_xlim(0, 9000)
    # 目盛りを非表示にする
    # ax.xaxis.set_visible(False)  # x軸の目盛りを非表示にする
    # ax.yaxis.set_visible(False)  # y軸の目盛りを非表示にする
    # 目盛りの位置と方向を調整
    ax.tick_params(which='both', direction='in', pad=10)
    ax.grid(True)
    
    if not os.path.exists(output_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"{game_id}_{subset_id}.png"),  bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)

def plot_vdep_by_event_with_event_type(vdep, game_id, subset_id, output_dir):
    # print(f"{game_id}_{subset_id}")
    possession_team = []
    vdep_team_a = []
    vdep_team_b = []
    preds_proba_concede_team_a = []
    preds_proba_concede_team_b = []
    preds_proba_foul_team_a = []
    preds_proba_foul_team_b = []
    preds_proba_fast_break_team_a = []
    preds_proba_fast_break_team_b = []
    time_team_a = []
    time_team_b = []
    texts_team_a = []
    texts_team_b = []
    
    # plot vdep by events
    for index, row in vdep.iterrows():
        possession_team.append(row['possession_team'])
        if row['possession_team'] == 0:
            if row['event_type'] == 'possession_change':
                vdep_team_b.append(np.nan)
                preds_proba_concede_team_b.append(np.nan)
                preds_proba_foul_team_b.append(np.nan)
                preds_proba_fast_break_team_b.append(np.nan)
                time_team_b.append(row['start_time'])
                texts_team_b.append("")
            vdep_team_b.append(row['vdep'])
            preds_proba_concede_team_b.append(row['preds_proba_concede'])
            preds_proba_foul_team_b.append(row['preds_proba_foul'])
            preds_proba_fast_break_team_b.append(row['preds_proba_fast_break'])
            time_team_b.append(row['start_time'])
            texts_team_b.append(row['event_type'])
            
        else:
            if row['event_type'] == 'possession_change':
                vdep_team_a.append(np.nan)
                preds_proba_concede_team_a.append(np.nan)
                preds_proba_foul_team_a.append(np.nan)
                preds_proba_fast_break_team_a.append(np.nan)
                time_team_a.append(row['start_time'])
                texts_team_a.append("")
            vdep_team_a.append(row['vdep'])
            preds_proba_concede_team_a.append(row['preds_proba_concede'])
            preds_proba_foul_team_a.append(row['preds_proba_foul'])
            preds_proba_fast_break_team_a.append(row['preds_proba_fast_break'])
            time_team_a.append(row['start_time'])
            texts_team_a.append(row['event_type'])
    """ # デバッグ用のprint文を追加
    print("Team A VDEP:", vdep_team_a)
    print("Team B VDEP:", vdep_team_b)
    print("Time Team A:", time_team_a)
    print("Time Team B:", time_team_b) """

    # 目標のピクセルサイズとdpi
    width_px = 9600
    height_px = 720
    dpi=100
    # 図のサイズをインチで計算
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize)
    # 余白の調整
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    alpha = 0.8
    ax2 = ax.twinx()
    ax2.plot(time_team_a, preds_proba_concede_team_a, label="Concede", color=COLORS2[2], alpha=alpha)
    ax2.plot(time_team_a, preds_proba_foul_team_a, label="Foul", color=COLORS2[3], alpha=alpha)
    ax2.plot(time_team_a, preds_proba_fast_break_team_a, label="Fast break", color=COLORS2[4], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_concede_team_b, label="Concede", color=COLORS2[2], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_foul_team_b, label="Foul", color=COLORS2[3], alpha=alpha)
    ax2.plot(time_team_b, preds_proba_fast_break_team_b, label="Fast break", color=COLORS2[4], alpha=alpha)
    # ax2.set_ylabel('Event probability')
    # ax2.yaxis.set_visible(False)
    ax.plot(time_team_a, vdep_team_a, color=COLORS2[0], marker='o', lw=4)
    ax.plot(time_team_b, vdep_team_b, color=COLORS2[1], marker='o', lw=4)
    
    ax.plot([], [], label="Team A's VDEP", color=COLORS2[0])
    ax.plot([], [], label="Team B's VDEP", color=COLORS2[1])
    ax.plot([], [], label="Concede", color=COLORS2[2])
    ax.plot([], [], label="Foul", color=COLORS2[3])
    ax.plot([], [], label="Fast break", color=COLORS2[4])
    ax.set_xlim(0, 9000)
    # 目盛りを非表示にする
    # ax.xaxis.set_visible(False)  # x軸の目盛りを非表示にする
    # ax.yaxis.set_visible(False)  # y軸の目盛りを非表示にする
    # 目盛りの位置と方向を調整
    ax.tick_params(which='both', direction='in', pad=10)
    ax.grid(True)
    
    # イベントタイプのテキストを表示
    texts = []
    for i, txt in enumerate(texts_team_a):
        if np.isfinite(time_team_a[i]) and np.isfinite(vdep_team_a[i]):
            texts.append(ax.annotate(txt, (time_team_a[i], vdep_team_a[i]), fontsize=14, color="k"))
    for i, txt in enumerate(texts_team_b):
        if np.isfinite(time_team_b[i]) and np.isfinite(vdep_team_b[i]):
            texts.append(ax.annotate(txt, (time_team_b[i], vdep_team_b[i]), fontsize=14, color="k"))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5))

    if not os.path.exists(output_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"{game_id}_{subset_id}.png"),  bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)

def create_legend_image(output_path):
    # 凡例の要素を作成
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[0], edgecolor='black', label="Team A's VDEP"),
        mpatches.Patch(facecolor=COLORS[1], edgecolor='black', label="Team B's VDEP"),
        mpatches.Patch(facecolor=COLORS[2], edgecolor='black', label='Concede'),
        mpatches.Patch(facecolor=COLORS[3], edgecolor='black', label='Foul'),
        mpatches.Patch(facecolor=COLORS[4], edgecolor='black', label='Fast break'),
    ]

    # 図を作成して凡例を配置
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.axis('off')
    legend = ax.legend(handles=legend_elements, loc='center', frameon=False)

    # フォントサイズやスタイルの調整（必要に応じて）
    for text in legend.get_texts():
        text.set_fontsize(12)

    # 凡例画像を保存
    fig.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    
def animate_vdep_scroll(image, output_file, final_width=1280, final_height=240, fps=15, legend_image=None):
    # output_fileが存在しない場合はmakedirsを実行   
    output_dir = os.path.dirname(output_file)
    print(f"output_dir:{output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 高解像度の設定
    upscale_factor = 4  # 解像度を2倍に
    high_res_width = final_width * upscale_factor
    high_res_height = final_height * upscale_factor

    # 背景画像の読み込み
    background = cv2.imread(image)
    if background is None:
        raise ValueError(f"背景画像 '{image}' を読み込めませんでした。ファイルパスを確認してください。")

    # 背景画像をリサイズ（高解像度で）
    bg_height, bg_width = background.shape[:2]
    aspect_ratio = high_res_height / bg_height
    new_width = int(bg_width * aspect_ratio)
    background = cv2.resize(background, (new_width, high_res_height), interpolation=cv2.INTER_LANCZOS4)

    # 左右に余白を追加（高解像度で）
    print(f"high_res_width:{high_res_width}")
    padding_width = high_res_width // 2  # 左右の余白の幅
    padded_width = new_width + (2 * padding_width)
    padded_background = np.zeros((high_res_height, padded_width, 3), dtype=np.uint8)
    padded_background[:, padding_width:padding_width + new_width] = background

    # 凡例画像の読み込みと処理（必要な場合のみ）
    if legend_image:
        legend = cv2.imread(legend_image, cv2.IMREAD_UNCHANGED)  # 透過情報を含めて読み込む
        if legend is None:
            raise ValueError(f"凡例画像 '{legend_image}' を読み込めませんでした。ファイルパスを確認してください。")
        
        legend_height, legend_width = legend.shape[:2]
        legend_aspect_ratio = legend_height / legend_width
        legend_display_width = int(high_res_width * 0.2)
        legend_display_height = int(legend_display_width * legend_aspect_ratio)
        legend_resized = cv2.resize(legend, (legend_display_width, legend_display_height), interpolation=cv2.INTER_AREA)

        # アルファチャンネルの処理
        if legend_resized.shape[2] == 4:
            alpha_channel = legend_resized[:, :, 3] / 255.0
            legend_rgb = legend_resized[:, :, :3]
        else:
            alpha_channel = np.ones((legend_display_height, legend_display_width), dtype=np.float32)
            legend_rgb = legend_resized
    else:
        legend_resized = None
        alpha_channel = None
        legend_rgb = None

    # 動画の長さ（秒）
    total_time = 300  # 5分
    total_frames = int(total_time * fps)

    # スクロールするピクセル数を計算
    scroll_pixels = padded_width - high_res_width
    if scroll_pixels <= 0:
        raise ValueError("背景画像の幅がフレーム幅よりも小さいため、スクロールできません。")

    def make_frame(t):
        frame_idx = int(t * fps)
        shift_x = int(scroll_pixels * (frame_idx / total_frames))
        frame = padded_background[:, shift_x:shift_x + high_res_width].copy()

        # 棒を描画（アンチエイリアシングを有効にする）
        center_x = high_res_width // 2
        cv2.line(frame, (center_x, 0), (center_x, high_res_height), (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)

        # 凡例がある場合のみ処理
        if legend_resized is not None:
            margin = 20 * upscale_factor  # マージンをスケーリング
            x_offset = high_res_width - legend_display_width - margin
            y_offset = margin
            y1, y2 = y_offset, y_offset + legend_display_height
            x1, x2 = x_offset, x_offset + legend_display_width

            # フレームに凡例を重ね合わせる
            frame_region = frame[y1:y2, x1:x2]
            alpha = alpha_channel[:, :, np.newaxis]
            frame[y1:y2, x1:x2] = alpha * legend_rgb + (1 - alpha) * frame_region

        # 最終的な解像度にリサイズ（高品質な補間を使用）
        frame_resized = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # 動画クリップを生成して書き出し
    animation = VideoClip(make_frame, duration=total_time)
    animation.write_videofile(output_file, fps=fps, codec='libx264', bitrate='800k')

def concat_videos_vertically(video_paths, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    clips = [VideoFileClip(path).set_fps(30) for path in video_paths]
    min_width = min(clip.w for clip in clips)
    clips_resized = [clip.resize(width=min_width) for clip in clips]
    final_clip = clips_array([[clip] for clip in clips_resized])
    final_clip = final_clip.resize(height=min(1080, final_clip.h))
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', bitrate='800k')

def concat_videos_horizontally(video_paths, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    clips = [VideoFileClip(path).set_fps(30) for path in video_paths]
    min_height = min(clip.h for clip in clips)
    clips_resized = [clip.resize(height=min_height) for clip in clips]
    final_clip = clips_array([clips_resized])
    final_clip = final_clip.resize(width=min(1920, final_clip.w))
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', bitrate='800k')

def overlay_videos(main_video_path, overlay_video_path, output_path, overlay_scale=0.3, offset_x=20, offset_y=20):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}. Please create it or provide a valid path.")
    main_cap = cv2.VideoCapture(main_video_path)
    overlay_cap = cv2.VideoCapture(overlay_video_path)
    width = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = main_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    overlay_width = int(overlay_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    overlay_height = int(overlay_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_overlay_width = int(overlay_width * overlay_scale)
    new_overlay_height = int(overlay_height * overlay_scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret_main, main_frame = main_cap.read()
        ret_overlay, overlay_frame = overlay_cap.read()
        if not ret_main or not ret_overlay:
            break
        overlay_frame_resized = cv2.resize(overlay_frame, (new_overlay_width, new_overlay_height))
        x_start = width - new_overlay_width - offset_x
        y_start = offset_y
        x_end = x_start + new_overlay_width
        y_end = y_start + new_overlay_height
        main_frame[y_start:y_end, x_start:x_end] = overlay_frame_resized
        out.write(main_frame)
    main_cap.release()
    overlay_cap.release()
    out.release()
    print(f"Video saved to {output_path}")

""" def plot_f1_scores(df, output_dir):
    # Calculate the mean for each of the F1 score columns
    mean_f1_concede = df['f1_concede'].mean()
    mean_f1_foul = df['f1_foul'].mean()
    mean_f1_fast_break = df['f1_fast_break'].mean()
    print(f"Mean F1 score for Concede: {mean_f1_concede}")
    print(f"Mean F1 score for Foul: {mean_f1_foul}")
    print(f"Mean F1 score for Fast break: {mean_f1_fast_break}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylabel("F1", fontsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.tight_layout()
    left = np.array([1, 2, 3])
    height = np.array([mean_f1_concede, mean_f1_foul, mean_f1_fast_break])
    label = ["Concede", "Foul", "Fast break"]
    rect = ax.bar(left, height, tick_label=label, align="center", color=COLORS[0], width=0.6)
    rect[1].set_color(COLORS[1])
    rect[2].set_color(COLORS[2])
    fig.savefig(output_dir) """
    
def plot_game_f1_scores(game_df, game_id, output_dir):
    """Plot F1 scores for a single game with error bars"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylabel("F1", fontsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    
    left = np.array([1, 2, 3])
    height = np.array([
        game_df['f1_concede'].iloc[0],
        game_df['f1_foul'].iloc[0],
        game_df['f1_fast_break'].iloc[0]
    ])
    yerr = np.array([
        game_df['std_concede'].iloc[0],
        game_df['std_foul'].iloc[0],
        game_df['std_fast_break'].iloc[0]
    ])
    label = ["Concede", "Foul", "Fast break"]
    
    rect = ax.bar(left, height, yerr=yerr, tick_label=label, 
                 align="center", color=COLORS[0], width=0.6,
                 capsize=10, ecolor='black', error_kw={'capthick': 2})
    
    rect[1].set_color(COLORS[1])
    rect[2].set_color(COLORS[2])
    plt.title(f"Game ID: {game_id}", fontsize=20)
    plt.tight_layout()
    
    fig.savefig(os.path.join(output_dir, f"f1_scores_game_{game_id}.png"))
    plt.close()

def plot_f1_scores(df, output_dir):
    # Create output directories
    overall_dir = output_dir
    by_game_dir = os.path.join(output_dir, "f1_scores_by_game")
    game_plots_dir = os.path.join(output_dir, "f1_scores_plots_by_game")
    os.makedirs(by_game_dir, exist_ok=True)
    os.makedirs(game_plots_dir, exist_ok=True)
    
    # Save and plot F1 scores by game
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]
        game_scores = {
            'game_id': game_id,
            'f1_concede': game_df['f1_concede'].mean(),
            'f1_foul': game_df['f1_foul'].mean(),
            'f1_fast_break': game_df['f1_fast_break'].mean(),
            'std_concede': game_df['f1_concede'].std(),
            'std_foul': game_df['f1_foul'].std(),
            'std_fast_break': game_df['f1_fast_break'].std()
        }
        # Save game stats
        pd.DataFrame([game_scores]).to_csv(
            os.path.join(by_game_dir, f"{game_id}.csv"), 
            index=False
        )
        # Plot game stats
        plot_game_f1_scores(
            pd.DataFrame([game_scores]), 
            game_id, 
            game_plots_dir
        )
    
    # Calculate overall statistics
    mean_f1_concede = df['f1_concede'].mean()
    mean_f1_foul = df['f1_foul'].mean()
    mean_f1_fast_break = df['f1_fast_break'].mean()
    
    std_f1_concede = df['f1_concede'].std()
    std_f1_foul = df['f1_foul'].std()
    std_f1_fast_break = df['f1_fast_break'].std()
    
    # Save overall statistics
    overall_stats = {
        'metric': ['Concede', 'Foul', 'Fast break'],
        'mean': [mean_f1_concede, mean_f1_foul, mean_f1_fast_break],
        'std': [std_f1_concede, std_f1_foul, std_f1_fast_break]
    }
    pd.DataFrame(overall_stats).to_csv(
        os.path.join(overall_dir, "f1_scores_overall.csv"), 
        index=False
    )
    
    # Print statistics
    print(f"Mean F1 score for Concede: {mean_f1_concede:.3f} ± {std_f1_concede:.3f}")
    print(f"Mean F1 score for Foul: {mean_f1_foul:.3f} ± {std_f1_foul:.3f}")
    print(f"Mean F1 score for Fast break: {mean_f1_fast_break:.3f} ± {std_f1_fast_break:.3f}")
    
    # Plot with error bars
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylabel("F1", fontsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.tight_layout()
    
    left = np.array([1, 2, 3])
    height = np.array([mean_f1_concede, mean_f1_foul, mean_f1_fast_break])
    yerr = np.array([std_f1_concede, std_f1_foul, std_f1_fast_break])
    label = ["Concede", "Foul", "Fast break"]
    
    rect = ax.bar(left, height, yerr=yerr, tick_label=label, 
                 align="center", color=COLORS[0], width=0.6,
                 capsize=10, ecolor='black', error_kw={'capthick': 2})
    
    rect[1].set_color(COLORS[1])
    rect[2].set_color(COLORS[2])
    
    fig.savefig(os.path.join(output_dir, "f1_scores_plot.png"))

team_name_mapping = {
    "ec7a6a": ("TBV", "SCM"),
    "48dcd3": ("WET", "SGF"),
    "ad969d": ("LEI", "BER"),
    "e0e547": ("BHC", "MTM"),
    "e8a35a": ("HAN", "RNL")
}

keeper_save_rate = {
    "TBV": 0.292,
    "SCM": 0.297,
    "WET": 0.264,
    "SGF": 0.314,
    "LEI": 0.282,
    "BER": 0.299,
    "BHC": 0.272,
    "MTM": 0.276,
    "HAN": 0.289,
    "RNL": 0.301
}

# Add league average save rate calculation
LEAGUE_AVG_SAVE_RATE = sum(keeper_save_rate.values()) / len(keeper_save_rate)

def normalize_concede(concede_val, team_short_name):
    """Normalize concede value using keeper save rates"""
    if team_short_name not in keeper_save_rate:
        return concede_val
        
    team_save_rate = keeper_save_rate[team_short_name]
    # Adjust concede value as if team had league average save rate
    shots_on_target = concede_val / (1 - team_save_rate)
    normalized_concede = shots_on_target * (1 - LEAGUE_AVG_SAVE_RATE)
    
    return normalized_concede

def plot_scatter(x, y, x_label, y_label, filename, labels, output_dir):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=80, c=COLORS[0])
    texts = []
    for i in range(len(x)):
        # Increase fontsize for text
        texts.append(plt.text(x[i], y[i], labels[i], fontsize=12))
    adjust_text(texts,
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                expand_points=(2, 2),
                expand_text=(2, 2),
                force_text=1,
                force_points=1)
    # Increase fontsize for axis labels and ticks
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=12)

    if "Rank" in y_label:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    corr = pd.Series(x).corr(pd.Series(y))
    print(f"Correlation between {x_label} and {y_label}: {corr}")
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()
    return corr

short_to_full = {
    "SGF": "Flensburg-Handewitt",
    "SCM": "SC Magdeburg",
    "HAN": "Hannover-Burgdorf",
    "RNL": "Rhein-Neckar Lowen",
    "BER": "Fuchse Berlin",
    "MTM": "MT Melsungen",
    "LEI": "DHfK Leipzig",
    "WET": "HSG Wetzlar",
    "TBV": "TBV Lemgo",
    "BHC": "Bergischer HC"
}
    
def plot_vdep_relation(games, output_dir, normalize_flag=False):
    print("plot_vdep_relation")

    vdep = []
    concede = []
    foul_prob = []
    concede_prob = []
    fast_break_prob = []
    transformed_foul_prob = []

    xy_vdep_concede = []
    xy_concedeprob_concede = []
    xy_foulprob_concede = []
    xy_fastbreakprob_concede = []
    xy_transformed_foulprob_concede = []
    team_tags = []

    for game_id, team_data in games.items():
        print(f"Processing game: {game_id}")
        team_a_name, team_b_name = team_name_mapping.get(game_id, ("Unknown A", "Unknown B"))

        for side, scores in team_data.items():
            # ショートネームをフルネームに変換してチーム名リストへ追加
            short_name = team_a_name if side == "team_a" else team_b_name
            full_name = short_to_full.get(short_name, short_name)
            team_tags.append(full_name)
            this_vdep = scores.get("vdep", np.nan)
            this_concede = scores.get("concede", np.nan)
            # Normalize concede value
            if normalize_flag:
                this_concede = normalize_concede(this_concede, short_name)
            this_foul_prob = scores.get("foul_prob", np.nan)
            this_concede_prob = scores.get("concede_prob", np.nan)
            this_fast_break_prob = scores.get("fast_break_prob", np.nan)
            this_transformed_foul_prob = scores.get("transformed_foul_prob", np.nan)

            vdep.append(this_vdep)
            concede.append(this_concede)
            foul_prob.append(this_foul_prob)
            concede_prob.append(this_concede_prob)
            fast_break_prob.append(this_fast_break_prob)
            transformed_foul_prob.append(this_transformed_foul_prob)

            xy_vdep_concede.append((this_vdep, this_concede))
            xy_concedeprob_concede.append((this_concede_prob, this_concede))
            xy_foulprob_concede.append((this_foul_prob, this_concede))
            xy_fastbreakprob_concede.append((this_fast_break_prob, this_concede))
            xy_transformed_foulprob_concede.append((this_transformed_foul_prob, this_concede))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    correlations = {}

    # 1) VDEP vs Concede
    x_vals_vdep = [x for (x, _) in xy_vdep_concede]
    y_vals_vdep = [y for (_, y) in xy_vdep_concede]
    correlations["vdep_vs_concede"] = plot_scatter(
        x_vals_vdep, y_vals_vdep, "H-VDEP", "Concede",
        "vdep_vs_concede.png", team_tags, output_dir
    )

    # 2) Concede Probability vs Concede
    x_vals_cp = [x for (x, _) in xy_concedeprob_concede]
    y_vals_cp = [y for (_, y) in xy_concedeprob_concede]
    correlations["concede_prob_vs_concede"] = plot_scatter(
        x_vals_cp, y_vals_cp, "Concede Probability", "Concede",
        "concede_prob_vs_concede.png", team_tags, output_dir
    )

    # 3) Foul Probability vs Concede
    x_vals_fp = [x for (x, _) in xy_foulprob_concede]
    y_vals_fp = [y for (_, y) in xy_foulprob_concede]
    correlations["foul_prob_vs_concede"] = plot_scatter(
        x_vals_fp, y_vals_fp, "Foul Probability", "Concede",
        "foul_prob_vs_concede.png", team_tags, output_dir
    )

    # 4) Fast Break Probability vs Concede
    x_vals_fb = [x for (x, _) in xy_fastbreakprob_concede]
    y_vals_fb = [y for (_, y) in xy_fastbreakprob_concede]
    correlations["fast_break_prob_vs_concede"] = plot_scatter(
        x_vals_fb, y_vals_fb, "Fast Break Probability", "Concede",
        "fast_break_prob_vs_concede.png", team_tags, output_dir
    )

    # 5) Transformed Foul Probability vs Concede
    x_vals_tf = [x for (x, _) in xy_transformed_foulprob_concede]
    y_vals_tf = [y for (_, y) in xy_transformed_foulprob_concede]
    correlations["transformed_foul_prob_vs_concede"] = plot_scatter(
        x_vals_tf, y_vals_tf, "Transformed Foul Probability", "Concede",
        "transformed_foul_prob_vs_concede.png", team_tags, output_dir
    )

    # 相関係数をCSVファイルとして保存
    corr_df = pd.DataFrame(list(correlations.items()), columns=["Comparison", "Correlation"])
    corr_df.to_csv("data/csv_files/correlations.csv", index=False)

    # 相関関係のヒートマップを作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.set_index("Comparison").T, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    print(f"Saved plots and correlation data to {output_dir}")
    
def plot_vdep_relation_season(games, csv_path, output_dir, normalize_flag=False):
    print("plot_vdep_relation_season")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # チーム名をフルネームで扱うため、season_statsとは別にshort_name→full_nameの対応を保持
    def load_season_stats(csv_path):
        df = pd.read_csv(csv_path)
        season_stats = {}
        short_to_full = {}
        for _, row in df.iterrows():
            team_name_full = row["Team"]
            short_name = row["Short Name"]
            matches = row["Matches"]
            goals_against = row["Goals Against"] / matches if matches else np.nan
            # Normalize goals against
            if normalize_flag:
                goals_against = normalize_concede(goals_against, short_name)
            
            # Short Nameをキーにして各種指標を格納
            season_stats[short_name] = {
                "rank": row["Rank"],
                "points": row["Points"],
                "matches": matches,
                "wins": row["Wins"],
                "draws": row["Draws"],
                "losses": row["Losses"],
                "goals_for": row["Goals For"] / matches if matches else np.nan,
                "goals_against": goals_against,
                "goal_diff": row["Goal Differenc"] / matches if matches else np.nan
            }
            # Short Name→フルネーム
            short_to_full[short_name] = team_name_full
        return season_stats, short_to_full

    season_stats, short_to_full = load_season_stats(csv_path)

    # ② gamesのキー(game_id)に対してteam_name_mappingを使って"短縮名"を取得。そこからフルネームに変換。
    teams = []
    vdep_vals = []
    foul_probs = []
    concede_probs = []
    fast_break_probs = []
    transformed_foul_probs = []

    for game_id, team_scores in games.items():
        tm_a, tm_b = team_name_mapping.get(game_id, ("Unknown A", "Unknown B"))
        for side, scores in team_scores.items():
            short_name = tm_a if side == "team_a" else tm_b
            # short_nameからフルネームに変換
            team_full_name = short_to_full.get(short_name, short_name)
            vdep_vals.append(scores.get("vdep", np.nan))
            foul_probs.append(scores.get("foul_prob", np.nan))
            concede_probs.append(scores.get("concede_prob", np.nan))
            fast_break_probs.append(scores.get("fast_break_prob", np.nan))
            transformed_foul_probs.append(scores.get("transformed_foul_prob", np.nan))
            teams.append(team_full_name)

    print(f"Teams (full names): {teams}")

    df = pd.DataFrame({
        "team": teams,
        "vdep": vdep_vals,
        "foul_prob": foul_probs,
        "concede_prob": concede_probs,
        "fast_break_prob": fast_break_probs,
        "transformed_foul_prob": transformed_foul_probs
    })
    # フルネームをキーに持たないため、ランクなどはshort_to_fullで逆引き不可なので、
    # ここでは short_name を使う代わりに、フルネーム→short_nameをつくってmapで引くか、
    # シンプルにshort_nameをdfに持たせる方法をとる。以下では逆マップを作成。
    # ただし初期にshort_to_fullを作るのと別にfull_to_shortを作る。
    full_to_short = {f: s for s, f in short_to_full.items()}

    def get_stat_value(team_full_name, key):
        short_ = full_to_short.get(team_full_name, "")
        return season_stats.get(short_, {}).get(key, np.nan)

    df["rank"] = df["team"].apply(lambda x: get_stat_value(x, "rank"))
    df["goals_for"] = df["team"].apply(lambda x: get_stat_value(x, "goals_for"))
    df["goals_against"] = df["team"].apply(lambda x: get_stat_value(x, "goals_against"))
    df["goal_diff"] = df["team"].apply(lambda x: get_stat_value(x, "goal_diff"))

    # ④ プロットなど従来の処理
    correlations = {}
    correlations["vdep"] = {
        "rank": plot_scatter(df["vdep"], df["rank"], "H-VDEP", "Season Rank", "vdep_vs_rank.png", df["team"], output_dir),
        "goals_for": plot_scatter(df["vdep"], df["goals_for"], "H-VDEP", "Season Goals For", "vdep_vs_goals_for.png", df["team"], output_dir),
        "goals_against": plot_scatter(df["vdep"], df["goals_against"], "H-VDEP", "Season Goals Against", "vdep_vs_goals_against.png", df["team"], output_dir),
        "goal_diff": plot_scatter(df["vdep"], df["goal_diff"], "H-VDEP", "Season Goal Diff", "vdep_vs_goal_diff.png", df["team"], output_dir)
    }
    correlations["foul_prob"] = {
        "rank": plot_scatter(df["foul_prob"], df["rank"], "Foul Probability", "Season Rank", "foul_prob_vs_rank.png", df["team"], output_dir),
        "goals_for": plot_scatter(df["foul_prob"], df["goals_for"], "Foul Probability", "Season Goals For", "foul_prob_vs_goals_for.png", df["team"], output_dir),
        "goals_against": plot_scatter(df["foul_prob"], df["goals_against"], "Foul Probability", "Season Goals Against", "foul_prob_vs_goals_against.png", df["team"], output_dir),
        "goal_diff": plot_scatter(df["foul_prob"], df["goal_diff"], "Foul Probability", "Season Goal Diff", "foul_prob_vs_goal_diff.png", df["team"], output_dir)
    }
    correlations["concede_prob"] = {
        "rank": plot_scatter(df["concede_prob"], df["rank"], "Concede Probability", "Season Rank", "concede_prob_vs_rank.png", df["team"], output_dir),
        "goals_for": plot_scatter(df["concede_prob"], df["goals_for"], "Concede Probability", "Season Goals For", "concede_prob_vs_goals_for.png", df["team"], output_dir),
        "goals_against": plot_scatter(df["concede_prob"], df["goals_against"], "Concede Probability", "Season Goals Against", "concede_prob_vs_goals_against.png", df["team"], output_dir),
        "goal_diff": plot_scatter(df["concede_prob"], df["goal_diff"], "Concede Probability", "Season Goal Diff", "concede_prob_vs_goal_diff.png", df["team"], output_dir)
    }
    correlations["fast_break_prob"] = {
        "rank": plot_scatter(df["fast_break_prob"], df["rank"], "Fast Break Probability", "Season Rank", "fast_break_prob_vs_rank.png", df["team"], output_dir),
        "goals_for": plot_scatter(df["fast_break_prob"], df["goals_for"], "Fast Break Probability", "Season Goals For", "fast_break_prob_vs_goals_for.png", df["team"], output_dir),
        "goals_against": plot_scatter(df["fast_break_prob"], df["goals_against"], "Fast Break Probability", "Season Goals Against", "fast_break_prob_vs_goals_against.png", df["team"], output_dir),
        "goal_diff": plot_scatter(df["fast_break_prob"], df["goal_diff"], "Fast Break Probability", "Season Goal Diff", "fast_break_prob_vs_goal_diff.png", df["team"], output_dir)
    }
    correlations["transformed_foul_prob"] = {
        "rank": plot_scatter(df["transformed_foul_prob"], df["rank"], "Transformed Foul Probability", "Season Rank", "transformed_foul_prob_vs_rank.png", df["team"], output_dir),
        "goals_for": plot_scatter(df["transformed_foul_prob"], df["goals_for"], "Transformed Foul Probability", "Season Goals For", "transformed_foul_prob_vs_goals_for.png", df["team"], output_dir),
        "goals_against": plot_scatter(df["transformed_foul_prob"], df["goals_against"], "Transformed Foul Probability", "Season Goals Against", "transformed_foul_prob_vs_goals_against.png", df["team"], output_dir),
        "goal_diff": plot_scatter(df["transformed_foul_prob"], df["goal_diff"], "Transformed Foul Probability", "Season Goal Diff", "transformed_foul_prob_vs_goal_diff.png", df["team"], output_dir)
    }

    corr_df = pd.DataFrame(correlations).T
    corr_df.to_csv(f"{output_dir}/correlations.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    print("Saved all season plots and correlation data to", output_dir)

# 1つのポゼッション内にファウルと失点があるかどうかの関係をプロット
def plot_concede_foul_relation():
    foul_flags = []
    concede_flags = []
    foul_flag_ = []
    concede_flag_ = []
    foul_flag = 0
    concede_flag = 0
    labeled_features_path = 'data/labeled_features_20241008'
    labeled_features_ = os.listdir(labeled_features_path)
    for i in tqdm(range(len(labeled_features_))):
        with open(os.path.join(labeled_features_path, labeled_features_[i]), 'rb') as f:
            labeled_features = pd.read_json(f)
        game_id = re.split('[_.]', labeled_features_[i])[2]
        subset_id = re.split('[_.]', labeled_features_[i])[3]
        for index, row in labeled_features.iterrows():
            if row['event_type'] == 'possession_change':
                foul_flag_.append(foul_flag)
                concede_flag_.append(concede_flag)
                foul_flag = 0
                concede_flag = 0
            if row['event_type'] == 'foul_free_throw':
                foul_flag = 1
            if row['event_type'] == 'goal':
                concede_flag = 1
        foul_flags.extend(foul_flag_)
        concede_flags.extend(concede_flag_)
        

    cm = confusion_matrix(foul_flags, concede_flags)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('foul_free_throw')
    plt.ylabel('goal')
    plt.savefig('data/fig/foul_concede_confusion_matrix.png')
    print(cm)
    print(chi2_contingency(np.array(cm)))
    
def animate_positions(h5_path, output_dir, fps=30, width=800, height=400):
    # HDF5ファイルを処理する
    for position_file in os.listdir(h5_path):
        if not position_file.endswith(".h5"):
            continue
        
        # HDF5ファイルのパス
        file_path = os.path.join(h5_path, position_file)
        output_file = os.path.join(output_dir, position_file.replace(".h5", ".mp4"))

        # HDF5ファイルを開く
        with h5py.File(file_path, 'r') as h5:
            balls = h5['balls'][:]
            team_a = h5['team_a'][:]
            team_b = h5['team_b'][:]

        # 動画作成用の設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # 描画用のプロット設定
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(0.0, 20.0)
        ax.add_patch(patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0))
        ax.add_patch(patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0))
        ax.add_patch(patches.Arc(xy=(40.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=90.0, theta2=180.0))
        ax.add_patch(patches.Arc(xy=(40.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=180.0, theta2=270.0))
        ax.plot([6.0, 6.0], [8.5, 11.5], color="k")
        ax.plot([34.0, 34.0], [8.5, 11.5], color="k")
        ax.plot([20.0, 20.0], [0.0, 20.0], color="k")

        # プロットオブジェクトを初期化
        ball_plot = ax.scatter([], [], c="y", label="Ball")
        team_a_plot = ax.scatter([], [], c="k", label="Team A")
        team_b_plot = ax.scatter([], [], c="r", label="Team B")

        # 凡例を追加
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

        # フレームごとの処理
        for frame_idx in tqdm(range(len(balls)), desc=f"Processing {position_file}"):
            # 既存のテキストを削除
            for text in ax.texts:
                text.remove()

            # ボールの位置を更新
            valid_ball_positions = np.array([pos for pos in balls[frame_idx] if not np.all(np.isnan(pos))])
            if valid_ball_positions.size > 0:
                if valid_ball_positions.ndim == 1:
                    valid_ball_positions = valid_ball_positions.reshape(1, -1)
                ball_plot.set_offsets(valid_ball_positions)
            else:
                ball_plot.set_offsets(np.empty((0, 2)))

            # Team Aの位置を更新
            valid_team_a_positions = np.array([pos for pos in team_a[frame_idx] if not np.all(np.isnan(pos))])
            if valid_team_a_positions.size > 0:
                if valid_team_a_positions.ndim == 1:
                    valid_team_a_positions = valid_team_a_positions.reshape(1, -1)
                team_a_plot.set_offsets(valid_team_a_positions)
                for idx, pos in enumerate(valid_team_a_positions):
                    ax.text(pos[0] + 0.3, pos[1] + 0.3, str(idx), fontsize=8, color='k')
            else:
                team_a_plot.set_offsets(np.empty((0, 2)))

            # Team Bの位置を更新
            valid_team_b_positions = np.array([pos for pos in team_b[frame_idx] if not np.all(np.isnan(pos))])
            if valid_team_b_positions.size > 0:
                if valid_team_b_positions.ndim == 1:
                    valid_team_b_positions = valid_team_b_positions.reshape(1, -1)
                team_b_plot.set_offsets(valid_team_b_positions)
                for idx, pos in enumerate(valid_team_b_positions):
                    ax.text(pos[0] + 0.3, pos[1] + 0.3, str(idx), fontsize=8, color='r')
            else:
                team_b_plot.set_offsets(np.empty((0, 2)))

            # 描画をキャンバスに転送
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            # フレームを動画に書き込む
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # 動画ファイルを閉じる
        video_writer.release()
        plt.close(fig)

        print(f"Animation saved to {output_file}")
