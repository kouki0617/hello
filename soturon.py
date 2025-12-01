import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 日本語フォントの設定（お使いの環境に合わせてフォント名を変更してください）
# 例: Windows -> 'Yu Gothic', Mac -> 'Hiragino Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止

# --------------------------------------------------------------------------
# アテンションダイナミクスのモデルを定義する関数
# --------------------------------------------------------------------------
def attention_dynamics(t, y, r1, r2, K1, K2, w12, w21, zeta1, zeta2):
    """
    アテンションダイナミクスの連立微分方程式
    y: [a1, a2, b1, b2] の状態ベクトル
    """
    a1, a2, b1, b2 = y
    
    # da1/dt
    da1_dt = r1 * a1 * (1 - (b1 + w12 * a2) / K1)
    
    # da2/dt
    da2_dt = r2 * a2 * (1 - (b2 + w21 * a1) / K2)
    
    # db1/dt
    db1_dt = a1 - zeta1 * b1
    
    # db2/dt
    db2_dt = a2 - zeta2 * b2
    
    return [da1_dt, da2_dt, db1_dt, db2_dt]

def calculate_K_ratio(w11, w12, w21, w22):
    """
    競争度行列の要素からK1/K2を計算
    K1/K2 = sqrt(w_21*w_22/w_12*w_11)
    """
    K_ratio = np.sqrt((w21 * w22) / (w12 * w11))
    return K_ratio

def event_extinction_a1(t, y, *params):
    return y[0] - 1e-3  # 種1の注目度が閾値を下回るとき0
event_extinction_a1.terminal = True
event_extinction_a1.direction = -1  # 減少方向で発火

def event_extinction_a2(t, y, *params):
    return y[1] - 1e-3
event_extinction_a2.terminal = True
event_extinction_a2.direction = -1

# --------------------------------------------------------------------------
# 共通のシミュレーション設定
# --------------------------------------------------------------------------
# 時間設定
t_span = [0, 500]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# 初期値 [a1(0), a2(0), b1(0), b2(0)]
initial_conditions = [10, 10, 0, 0]

# 共通パラメータ
r1, r2 = 1.0,1.0
K1, K2 = 100,200
zeta1, zeta2 = 0.5, 0.5

# --------------------------------------------------------------------------
# シナリオ1: 共存ケース
# w12 < K1 / (zeta2 * K2)  (1.0 < 1.67) -> OK
# w21 < K2 / (zeta1 * K1)  (2.0 < 2.0)  -> OK-------------------------------
params_coexistence = {
    'r1': r1, 'r2': r2,
    'K1': K1, 'K2': K2,
    'w12': 1.0, 'w21': 1.0,  # 競争係数
    'zeta1': zeta1, 'zeta2': zeta2
}

# 微分方程式を解く
sol_coexistence = solve_ivp(
    attention_dynamics,
    t_span,
    initial_conditions,
    args=tuple(params_coexistence.values()),
    dense_output=True,
    t_eval=t_eval
)

# --------------------------------------------------------------------------
# シナリオ2: 競合的排除ケース
# w12 < K1 / (zeta2 * K2)  (1.0 < 1.67) -> OK
# w21 > K2 / (zeta1 * K1)  (2.8 > 2.0)  -> NG
# --------------------------------------------------------------------------
params_exclusion = {
    'r1': r1, 'r2': r2,
    'K1': K1, 'K2': K2,
    'w12': 0.97, 'w21': 0.95,  # 競争係数
    'zeta1': zeta1, 'zeta2': zeta2
}

# 微分方程式を解く
sol_exclusion = solve_ivp(
    attention_dynamics,
    t_span,
    initial_conditions,
    args=tuple(params_exclusion.values()),
    dense_output=True,
    t_eval=t_eval
)

# --------------------------------------------------------------------------
# wを0~1の範囲で5つに分けてシミュレーション
# --------------------------------------------------------------------------
w_list = [0.2, 0.95] 
results = []

# 競争度行列の対角成分
w11, w22 = 1.0, 1.0

for w in w_list:
    w12 = w
    w21 = w
    
    # ρとK1/K2を計算
    
    K_ratio = calculate_K_ratio(w11, w12, w21, w22)
    
    # パラメータ設定
    params = {
        'r1': r1, 'r2': r2,
        'K1': K1, 'K2': K2,
        'w12': w12, 'w21': w21,
        'zeta1': zeta1, 'zeta2': zeta2
    }
    
    # シミュレーション実行
    sol = solve_ivp(
        attention_dynamics,
        t_span,
        initial_conditions,
        args=tuple(params.values()),
        dense_output=True,
        t_eval=t_eval
    )
    
    # 共存判定
    a1_final = sol.y[0][-1]
    a2_final = sol.y[1][-1]
    threshold = 1e-3
    is_coexistence = (a1_final > threshold) and (a2_final > threshold)
    
    # 絶滅時刻の計算（最初に閾値を下回った時刻、なければNaN）
    extinction_threshold = threshold
    a1_below = np.where(sol.y[0] <= extinction_threshold)[0]
    a2_below = np.where(sol.y[1] <= extinction_threshold)[0]
    extinction_time_a1 = sol.t[a1_below[0]] if a1_below.size > 0 else np.nan
    extinction_time_a2 = sol.t[a2_below[0]] if a2_below.size > 0 else np.nan
    
    # 結果をresultsリストに追加
    results.append((w, sol, 0, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2))
    



# --------------------------------------------------------------------------
# 2. w値別の注目度変化比較図
# --------------------------------------------------------------------------
# plt.figure(figsize=(12, 5))

# # サブプロット1: 複数w値での比較（種1）
# plt.subplot(1, 2, 1)
# # より幅広い色範囲を使用（0.1-0.9の範囲でより強いコントラスト）
# colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
# for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
#     linestyle = '-' if is_coexistence else '--'
#     linewidth = 3 if is_coexistence else 2  # 共存ケースをより太く
#     plt.plot(sol.t, sol.y[0], label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}', 
#              color=colors[i], linewidth=linewidth, linestyle=linestyle)
# plt.title('種1の注目度変化（w値別）')
# plt.xlabel('時間 t')
# plt.ylabel('注目度 $a_1(t)$')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # サブプロット2: 複数w値での比較（種2）
# plt.subplot(1, 2, 2)
# for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
#     linestyle = '-' if is_coexistence else '--'
#     linewidth = 3 if is_coexistence else 2  # 共存ケースをより太く
#     plt.plot(sol.t, sol.y[1], label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}', 
#              color=colors[i], linewidth=linewidth, linestyle=linestyle)
# plt.title('種2の注目度変化（w値別）')
# plt.xlabel('時間 t')
# plt.ylabel('注目度 $a_2(t)$')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.suptitle('w値別の注目度変化比較', fontsize=14)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------
# 3. w値別の詳細図
# --------------------------------------------------------------------------
# fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
# for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
#     ax = axes[i]
#     ax.plot(sol.t, sol.y[0], color='blue', linewidth=2)
#     ax.plot(sol.t, sol.y[1], color='red', linewidth=2)
#     status = "共存" if is_coexistence else "排除"
#     # wが小さい方に「競争が比較的小さい」、wが大きい方に「競争が比較的大きい」のタイトルを追加
#     if w < 0.6:  # wが小さい場合
#         ax.set_title('競争が比較的小さい')
#     else:  # wが大きい場合
#         ax.set_title('競争が比較的大きい')
#     ax.set_xlabel('時間 t')
#     ax.set_ylabel('注目度')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# # plt.suptitle('w値別の注目度変化詳細', fontsize=14)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------
# 3. wとζの関係図（w_listの変数を使用）
# --------------------------------------------------------------------------

zeta_range = np.linspace(0.1, 2.0, 50)
w_range = np.linspace(min(w_list), 1.0, 50)

# メッシュグリッドを作成
W, ZETA = np.meshgrid(w_range, zeta_range)


coexistence_condition = np.minimum(K1 / (ZETA * K2), K2 / (ZETA * K1))
coexistence_mask = W < coexistence_condition

# --------------------------------------------------------------------------
# 4. wとζの共存関係を効率よく調べる機能
# --------------------------------------------------------------------------

def check_coexistence_stability(w12, w21, zeta1, zeta2, K1, K2, r1, r2, 
                               initial_conditions, t_span, t_eval):
    """
    数値シミュレーションで共存の安定性をチェック
    """
    try:
        params = {
            'r1': r1, 'r2': r2,
            'K1': K1, 'K2': K2,
            'w12': w12, 'w21': w21,
            'zeta1': zeta1, 'zeta2': zeta2
        }
        
        sol = solve_ivp(
            attention_dynamics,
            t_span,
            initial_conditions,
            args=tuple(params.values()),
            dense_output=True,
            t_eval=t_eval
        )
        
        # 最終時刻での値
        a1_final = sol.y[0][-1]
        a2_final = sol.y[1][-1]
        
        # 両方の種が正の値で安定しているかチェック
        threshold = 1e-3
        is_coexistence = (a1_final > threshold) and (a2_final > threshold)
        
        return is_coexistence, a1_final, a2_final, sol
        
    except Exception as e:
        print(f"シミュレーションエラー: w12={w12}, w21={w21}, zeta1={zeta1}, zeta2={zeta2}")
        return False, 0, 0, None

def parameter_sweep_analysis(w_range, zeta_range, K1, K2, r1, r2, 
                            initial_conditions, t_span, t_eval):
    """
    パラメータスイープで共存領域を効率的に探索（絶滅時刻も記録）
    """
    print("パラメータスイープを開始...")
    
    # 結果を格納する配列
    w_values = []
    zeta_values = []
    coexistence_results = []
    a1_final_values = []
    a2_final_values = []
    extinction_times_a1 = []
    extinction_times_a2 = []
    
    total_combinations = len(w_range) * len(zeta_range)
    current = 0
    
    for w in w_range:
        for zeta in zeta_range:
            current += 1
            if current % 50 == 0:
                print(f"進行状況: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
            
            # 対称的な場合をテスト（w12 = w21 = w, zeta1 = zeta2 = zeta）
            is_coexistence, a1_final, a2_final, sol = check_coexistence_stability(
                w, w, zeta, zeta, K1, K2, r1, r2, initial_conditions, t_span, t_eval
            )
            
            # 絶滅時刻の計算
            extinction_threshold = 1e-3
            extinction_time_a1 = np.nan
            extinction_time_a2 = np.nan
            
            if sol is not None:
                a1_below = np.where(sol.y[0] <= extinction_threshold)[0]
                a2_below = np.where(sol.y[1] <= extinction_threshold)[0]
                extinction_time_a1 = sol.t[a1_below[0]] if a1_below.size > 0 else np.nan
                extinction_time_a2 = sol.t[a2_below[0]] if a2_below.size > 0 else np.nan
            
            w_values.append(w)
            zeta_values.append(zeta)
            coexistence_results.append(is_coexistence)
            a1_final_values.append(a1_final)
            a2_final_values.append(a2_final)
            extinction_times_a1.append(extinction_time_a1)
            extinction_times_a2.append(extinction_time_a2)
    
    return (np.array(w_values), np.array(zeta_values), 
            np.array(coexistence_results), np.array(a1_final_values), 
            np.array(a2_final_values), np.array(extinction_times_a1), 
            np.array(extinction_times_a2))

# パラメータ範囲を設定（ζ>1の場合も検証）
w_sweep = np.linspace(0.1, 1.0, 40)  # wの範囲を広げる
zeta_sweep = np.linspace(0.1, 2.0, 40)  # ζの範囲を0.1〜2.0に拡張

# パラメータスイープ実行
w_vals, zeta_vals, coexistence_mask, a1_finals, a2_finals, extinction_t1, extinction_t2 = parameter_sweep_analysis(
    w_sweep, zeta_sweep, K1, K2, r1, r2, initial_conditions, t_span, t_eval
)

# --------------------------------------------------------------------------
# 6. パラメータスイープ結果の可視化（絶滅時刻を含む）
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# 理論的共存条件との比較
# 理論的境界線を計算
# w=0を含める場合、理論的境界線の計算ではw > 0の条件が必要（0除算を避けるため）
w_theory = np.linspace(0, 1.0, 100)
# w=0の場合は理論的境界線を計算しない（w > 0の条件を追加）
w_theory_nonzero = w_theory[w_theory > 0]
zeta_boundary1 = K1 / (w_theory_nonzero * K2)
zeta_boundary2 = K2 / (w_theory_nonzero * K1)

# 有効な範囲のみプロット（ζ>1の場合も検証）
valid1 = (zeta_boundary1 >= 0.1) & (zeta_boundary1 <= 2.0)
valid2 = (zeta_boundary2 >= 0.1) & (zeta_boundary2 <= 2.0)

plt.plot(w_theory_nonzero[valid1], zeta_boundary1[valid1], 'b--', linewidth=2, 
         label=f'理論境界: w = K₁/(ζ×K₂)')
# plt.plot(w_theory_nonzero[valid2], zeta_boundary2[valid2], 'g--', linewidth=2, 
#          label=f'理論境界: w = K₂/(ζ×K₁)')

# 数値結果を重ね合わせ
plt.scatter(w_vals[coexistence_mask], zeta_vals[coexistence_mask], 
           color='green', alpha=0.6, s=40, label='数値: 共存可能')
plt.scatter(w_vals[~coexistence_mask], zeta_vals[~coexistence_mask], 
           color='red', alpha=0.6, s=40, label='数値: 共存不可能')

plt.xlabel('w (競争係数)')
plt.ylabel('ζ (減衰率)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 6.5. 絶滅時刻マップ（より広い色幅で強調表示）
# --------------------------------------------------------------------------
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# 1. 種1の絶滅時刻マップ
extinction_mask_a1 = ~np.isnan(extinction_t1)
if np.any(extinction_mask_a1):
    # 絶滅時刻の範囲を調整してコントラストを強化
    extinction_times_a1 = extinction_t1[extinction_mask_a1]
    vmin = np.percentile(extinction_times_a1, 5)  # 下位5%を最小値
    vmax = np.percentile(extinction_times_a1, 95)  # 上位5%を最大値
    
    scatter1 = ax1.scatter(w_vals[extinction_mask_a1], zeta_vals[extinction_mask_a1], 
                          c=extinction_times_a1, cmap='plasma', 
                          alpha=0.8, s=50, vmin=vmin, vmax=vmax)
    plt.colorbar(scatter1, ax=ax1, label='絶滅時刻', shrink=0.8)
ax1.set_xlabel('w (競争係数)')
ax1.set_ylabel('ζ (減衰率)')
ax1.set_title('種1の絶滅時刻マップ')
ax1.grid(True, alpha=0.3)

# 2. 種2の絶滅時刻マップ
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
extinction_mask_a2 = ~np.isnan(extinction_t2)
if np.any(extinction_mask_a2):
    extinction_times_a2 = extinction_t2[extinction_mask_a2]
    vmin = np.percentile(extinction_times_a2, 5)
    vmax = np.percentile(extinction_times_a2, 95)
    
    scatter2 = ax2.scatter(w_vals[extinction_mask_a2], zeta_vals[extinction_mask_a2], 
                          c=extinction_times_a2, cmap='viridis', 
                          alpha=0.8, s=50, vmin=vmin, vmax=vmax)
    plt.colorbar(scatter2, ax=ax2, label='絶滅時刻', shrink=0.8)
ax2.set_xlabel('w (競争係数)')
ax2.set_ylabel('ζ (減衰率)')
ax2.set_title('種2の絶滅時刻マップ')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 7. 絶滅時刻の統計分析（パラメータスイープ版）
# --------------------------------------------------------------------------
print("\n=== パラメータスイープ絶滅時刻分析 ===")

# 絶滅が発生したケースの統計
extinction_occurred = ~(np.isnan(extinction_t1) & np.isnan(extinction_t2))
extinction_rate = np.mean(extinction_occurred)
print(f"絶滅発生率: {extinction_rate*100:.1f}% ({np.sum(extinction_occurred)}/{len(extinction_occurred)} ケース)")

if np.any(extinction_occurred):
    # 種1の絶滅統計
    a1_extinct = ~np.isnan(extinction_t1)
    if np.any(a1_extinct):
        print(f"\n種1絶滅統計:")
        print(f"  発生率: {np.mean(a1_extinct)*100:.1f}%")
        print(f"  平均絶滅時刻: {np.nanmean(extinction_t1):.2f}")
        print(f"  絶滅時刻の標準偏差: {np.nanstd(extinction_t1):.2f}")
    
    # 種2の絶滅統計
    a2_extinct = ~np.isnan(extinction_t2)
    if np.any(a2_extinct):
        print(f"\n種2絶滅統計:")
        print(f"  発生率: {np.mean(a2_extinct)*100:.1f}%")
        print(f"  平均絶滅時刻: {np.nanmean(extinction_t2):.2f}")
        print(f"  絶滅時刻の標準偏差: {np.nanstd(extinction_t2):.2f}")
    
    


# 統計情報を表示
print("\n=== パラメータスイープ結果 ===")
print(f"総テスト数: {len(w_vals)}")
print(f"共存可能な組み合わせ: {np.sum(coexistence_mask)} ({np.mean(coexistence_mask)*100:.1f}%)")
print(f"共存不可能な組み合わせ: {np.sum(~coexistence_mask)} ({np.mean(~coexistence_mask)*100:.1f}%)")

# 最も安定した共存パラメータを特定
if np.any(coexistence_mask):
    stable_indices = coexistence_mask & (a1_finals > 5) & (a2_finals > 5)
    if np.any(stable_indices):
        best_idx = np.argmax(a1_finals[stable_indices] + a2_finals[stable_indices])
        best_w = w_vals[stable_indices][best_idx]
        best_zeta = zeta_vals[stable_indices][best_idx]
        print(f"\n最も安定した共存パラメータ:")
        print(f"w = {best_w:.3f}, ζ = {best_zeta:.3f}")
        print(f"最終注目度: a1 = {a1_finals[stable_indices][best_idx]:.3f}, a2 = {a2_finals[stable_indices][best_idx]:.3f}")

# --------------------------------------------------------------------------
# wとK2/K1の比での共存範囲分析
# --------------------------------------------------------------------------
def parameter_sweep_w_K2K1_ratio(w_range, K2K1_ratio_range, K1_fixed, zeta1, zeta2, 
                                 r1, r2, initial_conditions, t_span, t_eval):
    
    print("wとK2/K1の比のパラメータスイープを開始...")
    
    # 結果を格納する配列
    w_values = []
    K2K1_ratio_values = []
    coexistence_results = []
    a1_final_values = []
    a2_final_values = []
    
    total_combinations = len(w_range) * len(K2K1_ratio_range)
    current = 0
    
    for w in w_range:
        for K2K1_ratio in K2K1_ratio_range:
            current += 1
            if current % 50 == 0:
                print(f"進行状況: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
            
            # K2を計算（K1は固定）
            K2_val = K2K1_ratio * K1_fixed
            
            # 対称的な場合をテスト（w12 = w21 = w, zeta1 = zeta2）
            is_coexistence, a1_final, a2_final, sol = check_coexistence_stability(
                w, w, zeta1, zeta2, K1_fixed, K2_val, 
                r1, r2, initial_conditions, t_span, t_eval
            )
            
            w_values.append(w)
            K2K1_ratio_values.append(K2K1_ratio)
            coexistence_results.append(is_coexistence)
            a1_final_values.append(a1_final)
            a2_final_values.append(a2_final)
    
    return (np.array(w_values), np.array(K2K1_ratio_values), 
            np.array(coexistence_results), np.array(a1_final_values), 
            np.array(a2_final_values))


def plot_w_K2K1_coexistence(w_values, K2K1_ratio_values, coexistence_results, 
                            a1_final_values, a2_final_values, zeta1, zeta2, K1_fixed):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 共存判定マップ（2値）
    ax1 = axes[0]
    ax1.scatter(w_values[coexistence_results], 
               K2K1_ratio_values[coexistence_results], 
               c='green', alpha=0.6, s=20, label='共存可能')
    ax1.scatter(w_values[~coexistence_results], 
               K2K1_ratio_values[~coexistence_results], 
               c='red', alpha=0.6, s=20, label='共存不可能')
    
    # 理論的境界線を計算（x軸がw、y軸がK2/K1）
    w_theory = np.linspace(0.01, 1.0, 200)  # w=0を除外（0除算を避けるため）
    
    # 理論的境界線1: K2/K1 = w*ζ
    K2K1_boundary1 = w_theory * zeta1  # K2/K1 = w*ζ1
    ax1.plot(w_theory, K2K1_boundary1, 'b--', linewidth=2)
    
    # 理論的境界線2: K2/K1 = 1/(w*ζ)
    K2K1_boundary2 = 1.0 / (w_theory * zeta2)  # K2/K1 = 1/(w*ζ2)
    # 有効な範囲のみ描画（K2/K1が適切な範囲内）
    valid_boundary2 = (K2K1_boundary2 >= K2K1_ratio_values.min()) & \
                      (K2K1_boundary2 <= K2K1_ratio_values.max())
    if np.any(valid_boundary2):
        ax1.plot(w_theory[valid_boundary2], K2K1_boundary2[valid_boundary2], 
                'r--', linewidth=2)
    
    # 共存条件: w < K1/(zeta2*K2) かつ w < K2/(zeta1*K1)
    # K2/K1 = R とすると、w < 1/(zeta2*R) かつ w < R/(zeta1)
    # 各wに対して、共存可能なK2/K1の範囲を計算
    K2K1_boundary = []
    w_boundary = []
    for w_val in w_theory:
        if w_val > 0:
            # w < 1/(zeta2*R) より R < 1/(zeta2*w)
            R_max1 = 1.0 / (zeta2 * w_val)
            # w < R/zeta1 より R > w*zeta1
            R_min2 = w_val * zeta1
            # 両方の条件を満たす範囲（R_min2 < R < R_max1）
            if R_max1 > R_min2:
                # 境界はR_max1（上限）
                K2K1_boundary.append(R_max1)
                w_boundary.append(w_val)
    
    # 共存境界線を描画（オプション：既存の境界線と重複する場合はコメントアウト）
    # if len(K2K1_boundary) > 0:
    #     ax1.plot(w_boundary, K2K1_boundary, 'g--', linewidth=2, 
    #             label='共存境界: w = K₁/(ζ₂×K₂)')
    
    ax1.set_xlabel('w (競争係数)', fontsize=12)
    ax1.set_ylabel('K₂/K₁ 比', fontsize=12)
    ax1.set_title('wとK₂/K₁比による共存範囲マップ', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(K2K1_ratio_values.min(), K2K1_ratio_values.max())
    
    # 2. 最終注目度の合計による強度マップ
    ax2 = axes[1]
    total_attention = a1_final_values + a2_final_values
    coexistence_mask = coexistence_results & (total_attention > 1e-3)
    if np.any(coexistence_mask):
        scatter = ax2.scatter(w_values[coexistence_mask], 
                             K2K1_ratio_values[coexistence_mask], 
                             c=total_attention[coexistence_mask], 
                             cmap='viridis', alpha=0.8, s=20, 
                             vmin=np.percentile(total_attention[coexistence_mask], 5),
                             vmax=np.percentile(total_attention[coexistence_mask], 95))
        plt.colorbar(scatter, ax=ax2, label='最終注目度の合計 (a₁ + a₂)', shrink=0.8)
    ax2.scatter(w_values[~coexistence_results], 
               K2K1_ratio_values[~coexistence_results], 
               c='lightgray', alpha=0.3, s=20)
    ax2.set_xlabel('w (競争係数)', fontsize=12)
    ax2.set_ylabel('K₂/K₁ 比', fontsize=12)
    ax2.set_title('共存領域の安定性マップ（最終注目度の合計）', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(K2K1_ratio_values.min(), K2K1_ratio_values.max())
    
    plt.tight_layout()
    plt.show()
    
    # 3. 統計情報を表示
    print("\n=== wとK₂/K₁比のパラメータスイープ結果 ===")
    print(f"総テスト数: {len(w_values)}")
    print(f"共存可能な組み合わせ: {np.sum(coexistence_results)} "
          f"({np.mean(coexistence_results)*100:.1f}%)")
    print(f"共存不可能な組み合わせ: {np.sum(~coexistence_results)} "
          f"({np.mean(~coexistence_results)*100:.1f}%)")
    
    # 最も安定した共存パラメータを特定
    if np.any(coexistence_results):
        stable_indices = coexistence_results & (a1_final_values > 5) & (a2_final_values > 5)
        if np.any(stable_indices):
            best_idx = np.argmax(a1_final_values[stable_indices] + a2_final_values[stable_indices])
            best_w = w_values[stable_indices][best_idx]
            best_K2K1 = K2K1_ratio_values[stable_indices][best_idx]
            print(f"\n最も安定した共存パラメータ:")
            print(f"w = {best_w:.3f}, K₂/K₁ = {best_K2K1:.3f}")
            print(f"最終注目度: a1 = {a1_final_values[stable_indices][best_idx]:.3f}, "
                  f"a2 = {a2_final_values[stable_indices][best_idx]:.3f}")


# wとK2/K1の比のパラメータスイープを実行
print("\n=== wとK₂/K₁比の共存範囲分析 ===")
K1_fixed = 100  # K1を固定
K2K1_ratio_range = np.linspace(0.1, 5.0, 50)  # K2/K1の比の範囲
w_range_K2K1 = np.linspace(0, 1.0, 50)  # wの範囲（0から開始可能）

print(f"K1 = {K1_fixed} に固定")
print(f"K₂/K₁比の範囲: {K2K1_ratio_range[0]:.2f} ～ {K2K1_ratio_range[-1]:.2f}")
print(f"wの範囲: {w_range_K2K1[0]:.2f} ～ {w_range_K2K1[-1]:.2f}")
print("シミュレーション実行中...")

w_vals_K2K1, K2K1_ratio_vals, coexistence_results_K2K1, \
    a1_final_K2K1, a2_final_K2K1 = parameter_sweep_w_K2K1_ratio(
        w_range_K2K1, K2K1_ratio_range, K1_fixed, zeta1, zeta2, 
        r1, r2, initial_conditions, t_span, t_eval
    )

# 結果を可視化
plot_w_K2K1_coexistence(w_vals_K2K1, K2K1_ratio_vals, coexistence_results_K2K1,
                        a1_final_K2K1, a2_final_K2K1, zeta1, zeta2, K1_fixed)

# --------------------------------------------------------------------------
# a2の注目度を最大化する最適なポジションの探索
# --------------------------------------------------------------------------
def parameter_sweep_w_K2_for_a2(w_range, K2_range, K1_fixed, zeta1, zeta2, 
                                r1, r2, initial_conditions, t_span, t_eval):
    """
    wとK2の範囲でパラメータスイープを実行し、a2の最終注目度を計算
    
    Parameters:
    -----------
    w_range : array-like
        w（競争係数）の範囲
    K2_range : array-like
        K2（環境収容力）の範囲
    K1_fixed : float
        固定するK1の値
    zeta1, zeta2 : float
        減衰率
    r1, r2 : float
        成長率
    initial_conditions : array-like
        初期条件
    t_span : list
        時間範囲
    t_eval : array-like
        評価する時間点
    
    Returns:
    --------
    tuple
        (w_values, K2_values, a2_final_values, coexistence_results)
    """
    print("\n=== a2の注目度を最大化する最適なポジションの探索 ===")
    print(f"K1 = {K1_fixed} に固定")
    print(f"wの範囲: {w_range[0]:.2f} ～ {w_range[-1]:.2f}")
    print(f"K2の範囲: {K2_range[0]:.2f} ～ {K2_range[-1]:.2f}")
    print("シミュレーション実行中...")
    
    w_values = []
    K2_values = []
    a2_final_values = []
    coexistence_results = []
    
    total_combinations = len(w_range) * len(K2_range)
    current = 0
    
    for w in w_range:
        for K2_val in K2_range:
            current += 1
            if current % 100 == 0:
                print(f"進行状況: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
            
            # 対称的な場合をテスト（w12 = w21 = w）
            is_coexistence, a1_final, a2_final, sol = check_coexistence_stability(
                w, w, zeta1, zeta2, K1_fixed, K2_val, 
                r1, r2, initial_conditions, t_span, t_eval
            )
            
            w_values.append(w)
            K2_values.append(K2_val)
            a2_final_values.append(a2_final)
            coexistence_results.append(is_coexistence)
    
    return (np.array(w_values), np.array(K2_values), 
            np.array(a2_final_values), np.array(coexistence_results))


def plot_a2_heatmap(w_values, K2_values, a2_final_values, coexistence_results, 
                    K1_fixed, zeta1, zeta2):
    """
    a2の最終注目度をヒートマップで可視化
    
    Parameters:
    -----------
    w_values : array-like
        wの値の配列
    K2_values : array-like
        K2の値の配列
    a2_final_values : array-like
        a2の最終注目度の配列
    coexistence_results : array-like
        共存判定結果の配列（bool）
    K1_fixed : float
        固定したK1の値
    zeta1, zeta2 : float
        減衰率
    """
    # データをグリッドに変換
    w_unique = np.unique(w_values)
    K2_unique = np.unique(K2_values)
    w_grid, K2_grid = np.meshgrid(w_unique, K2_unique)
    
    # a2の値をグリッドに変換
    a2_grid = np.zeros_like(w_grid)
    for i, w_val in enumerate(w_unique):
        for j, K2_val in enumerate(K2_unique):
            idx = np.where((w_values == w_val) & (K2_values == K2_val))[0]
            if len(idx) > 0:
                a2_grid[j, i] = a2_final_values[idx[0]]
    
    # 共存領域のマスク
    coexistence_grid = np.zeros_like(w_grid, dtype=bool)
    for i, w_val in enumerate(w_unique):
        for j, K2_val in enumerate(K2_unique):
            idx = np.where((w_values == w_val) & (K2_values == K2_val))[0]
            if len(idx) > 0:
                coexistence_grid[j, i] = coexistence_results[idx[0]]
    
    # 図を作成
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # a2の最終注目度のヒートマップ（全体）
    im = ax.contourf(w_grid, K2_grid, a2_grid, levels=50, cmap='viridis')
    plt.colorbar(im, ax=ax, label='a₂の最終注目度', shrink=0.8)
    ax.set_xlabel('w (競争係数)', fontsize=12)
    ax.set_ylabel('K₂ (環境収容力)', fontsize=12)
    ax.set_title(f'a₂の最終注目度ヒートマップ (K₁={K1_fixed}固定)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 共存不可能な領域をグレーでマスク
    a2_masked = np.ma.masked_where(~coexistence_grid, a2_grid)
    ax.contourf(w_grid, K2_grid, a2_masked, levels=50, cmap='gray', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計情報を表示
    print("\n=== a₂の注目度分析結果 ===")
    print(f"総テスト数: {len(w_values)}")
    print(f"共存可能な組み合わせ: {np.sum(coexistence_results)} "
          f"({np.mean(coexistence_results)*100:.1f}%)")
    
    # 全体での最大値
    max_idx_flat = np.argmax(a2_final_values)
    print(f"\n【全体での最大値】")
    print(f"  w = {w_values[max_idx_flat]:.3f}")
    print(f"  K₂ = {K2_values[max_idx_flat]:.2f}")
    print(f"  a₂ = {a2_final_values[max_idx_flat]:.3f}")
    print(f"  共存: {'可能' if coexistence_results[max_idx_flat] else '不可能'}")


# a2の注目度を最大化する最適なポジションを探索
print("\n=== a₂の注目度を最大化する最適なポジションの探索 ===")
K1_fixed_a2 = 100  # K1を固定
w_range_a2 = np.linspace(0.01, 0.99, 50)  # wの範囲（0 < w < 1）
K2_range_a2 = np.linspace(1, 99, 50)  # K2の範囲（0 < K2 < 100）

w_vals_a2, K2_vals_a2, a2_final_vals, coexistence_results_a2 = \
    parameter_sweep_w_K2_for_a2(
        w_range_a2, K2_range_a2, K1_fixed_a2, zeta1, zeta2, 
        r1, r2, initial_conditions, t_span, t_eval
    )

# 結果を可視化
plot_a2_heatmap(w_vals_a2, K2_vals_a2, a2_final_vals, coexistence_results_a2,
                K1_fixed_a2, zeta1, zeta2)
