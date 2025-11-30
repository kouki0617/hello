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

# --------------------------------------------------------------------------
# ρを求める関数
# --------------------------------------------------------------------------
# def calculate_rho(w11, w12, w21, w22):
#     """
#     競争度行列の要素からρを計算
#     ρ = sqrt(w_12*w_21/w_11*w_22)
#     """
#     rho = np.sqrt((w12 * w21) / (w11 * w22))
#     return rho

# --------------------------------------------------------------------------
# K1/K2を求める関数
# --------------------------------------------------------------------------
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
zeta1, zeta2 = 0.8, 0.8

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
# 1. 基本ケースの比較図
# --------------------------------------------------------------------------
# plt.figure(figsize=(12, 5))

# # サブプロット1: 共存ケース
# plt.subplot(1, 2, 1)
# plt.plot(sol_coexistence.t, sol_coexistence.y[0], label='種1 注目度 $a_1(t)$', color='blue', linewidth=2)
# plt.plot(sol_coexistence.t, sol_coexistence.y[1], label='種2 注目度 $a_2(t)$', color='red', linewidth=2)
# plt.title('共存ケース (w12=w21=1.0)')
# plt.xlabel('時間 t')
# plt.ylabel('注目度')
# plt.legend()
# plt.grid(True, alpha=0.3)

# サブプロット2: 競合的排除ケース
# plt.subplot(1, 2, 2)
# plt.plot(sol_exclusion.t, sol_exclusion.y[0], label='種1 注目度 $a_1(t)$', color='blue', linewidth=2)
# plt.plot(sol_exclusion.t, sol_exclusion.y[1], label='種2 注目度 $a_2(t)$', color='red', linewidth=2)
# plt.title('競合的排除ケース (w12=w21=0.97)')
# plt.xlabel('時間 t')
# plt.ylabel('注目度')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.suptitle('基本ケースの比較', fontsize=14)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------
# 2. w値別の注目度変化比較図
# --------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# サブプロット1: 複数w値での比較（種1）
plt.subplot(1, 2, 1)
# より幅広い色範囲を使用（0.1-0.9の範囲でより強いコントラスト）
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
    linestyle = '-' if is_coexistence else '--'
    linewidth = 3 if is_coexistence else 2  # 共存ケースをより太く
    plt.plot(sol.t, sol.y[0], label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}', 
             color=colors[i], linewidth=linewidth, linestyle=linestyle)
plt.title('種1の注目度変化（w値別）')
plt.xlabel('時間 t')
plt.ylabel('注目度 $a_1(t)$')
plt.legend()
plt.grid(True, alpha=0.3)

# サブプロット2: 複数w値での比較（種2）
plt.subplot(1, 2, 2)
for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
    linestyle = '-' if is_coexistence else '--'
    linewidth = 3 if is_coexistence else 2  # 共存ケースをより太く
    plt.plot(sol.t, sol.y[1], label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}', 
             color=colors[i], linewidth=linewidth, linestyle=linestyle)
plt.title('種2の注目度変化（w値別）')
plt.xlabel('時間 t')
plt.ylabel('注目度 $a_2(t)$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('w値別の注目度変化比較', fontsize=14)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 3. w値別の詳細図
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
for i, (w, sol, rho, K_ratio, is_coexistence, a1_final, a2_final, extinction_time_a1, extinction_time_a2) in enumerate(results):
    ax = axes[i]
    ax.plot(sol.t, sol.y[0], color='blue', linewidth=2)
    ax.plot(sol.t, sol.y[1], color='red', linewidth=2)
    status = "共存" if is_coexistence else "排除"
    # wが小さい方に「競争が比較的小さい」、wが大きい方に「競争が比較的大きい」のタイトルを追加
    if w < 0.6:  # wが小さい場合
        ax.set_title('競争が比較的小さい')
    else:  # wが大きい場合
        ax.set_title('競争が比較的大きい')
    ax.set_xlabel('時間 t')
    ax.set_ylabel('注目度')
    ax.legend()
    ax.grid(True, alpha=0.3)

# plt.suptitle('w値別の注目度変化詳細', fontsize=14)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 3. wとζの関係図（w_listの変数を使用）
# --------------------------------------------------------------------------
# ζの範囲を設定（0~2の範囲に拡張してζ>1の場合を検証）

# ζ=0 を避けてゼロ除算を防ぐ
zeta_range = np.linspace(0.1, 2.0, 50)
w_range = np.linspace(min(w_list), 1.0, 50)

# メッシュグリッドを作成
W, ZETA = np.meshgrid(w_range, zeta_range)

# 共存条件を計算
# w12 < K1 / (zeta2 * K2) かつ w21 < K2 / (zeta1 * K1)
# w12 = w21 = w, zeta1 = zeta2 = zeta の場合
# w < K1 / (zeta * K2) かつ w < K2 / (zeta * K1)
# w < min(K1/(zeta*K2), K2/(zeta*K1))

coexistence_condition = np.minimum(K1 / (ZETA * K2), K2 / (ZETA * K1))
coexistence_mask = W < coexistence_condition

# --------------------------------------------------------------------------
# 4. wとζの共存条件図（コメントアウト）
# -------------------------------------------------------ValueError: too many values to unpack (expected 7)

# plt.figure(figsize=(12, 8))
# # 共存可能領域を塗りつぶし
# plt.contourf(W, ZETA, coexistence_mask.astype(int), levels=[0, 0.5, 1], 
#              colors=['lightcoral', 'lightgreen'], alpha=0.6)
# plt.contour(W, ZETA, coexistence_mask.astype(int), levels=[0.5], 
#             colors=['black'], linewidths=2)

# # 各w値でのパラメータ点もプロット
# for w in w_list:
#     plt.scatter(w, zeta1, color='blue', s=50, alpha=0.7, zorder=4)
#     plt.annotate(f'w={w}', (w, zeta1), xytext=(5, 5), textcoords='offset points', fontsize=8)

# # 境界線の数式を表示
# w_boundary = np.linspace(min(w_list), 1.0, 100)
# zeta_boundary1 = K1 / (w_boundary * K2)  # w = K1/(zeta*K2) から zeta = K1/(w*K2)
# zeta_boundary2 = K2 / (w_boundary * K1)  # w = K2/(zeta*K1) から zeta = K2/(w*K1)

# # 有効な範囲のみプロット
# valid1 = (zeta_boundary1 >= 0) & (zeta_boundary1 <= 2.0)
# valid2 = (zeta_boundary2 >= 0) & (zeta_boundary2 <= 2.0)

# plt.plot(w_boundary[valid1], zeta_boundary1[valid1], 'b--', linewidth=2, 
#          label=f'w = K₁/(ζ×K₂) = {K1}/{K2}/ζ')
# plt.plot(w_boundary[valid2], zeta_boundary2[valid2], 'g--', linewidth=2, 
#          label=f'w = K₂/(ζ×K₁) = {K2}/{K1}/ζ')

# ax = plt.gca()
# # 軸の設定
# ax.set_xlabel('w (競争係数)', fontsize=14)
# ax.set_ylabel('ζ (減衰率)', fontsize=14)
# ax.set_title('wとζの関係：共存条件', fontsize=16)
# ax.grid(True, alpha=0.3)
# ax.legend()

# # 軸の範囲を設定（ζ>1の場合も検証）
# ax.set_xlim(min(w_list), 1)
# ax.set_ylim(0, 2)

# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------
# 5. w値別の共存率図
# --------------------------------------------------------------------------
# plt.figure(figsize=(8, 6))

# w_unique = np.unique([res[0] for res in results])
# coexistence_rates = []
# for w in w_unique:
#     # 該当するw値の結果を取得
#     w_results = [r for r in results if r[0] == w]
#     if w_results:
#         rate = np.mean([r[4] for r in w_results])  # is_coexistenceの平均
#         coexistence_rates.append(rate)
#     else:
#         coexistence_rates.append(0)

# plt.plot(w_unique, coexistence_rates, 'o-', linewidth=2, markersize=8)
# plt.xlabel('w (競争係数)', fontsize=14)
# plt.ylabel('共存率', fontsize=14)
# plt.title('w値別の共存率', fontsize=16)
# plt.grid(True, alpha=0.3)
# plt.ylim(0, 1)

# # 各点に値を表示
# for i, (w, rate) in enumerate(zip(w_unique, coexistence_rates)):
#     plt.annotate(f'{rate:.2f}', (w, rate), xytext=(0, 10), 
#                 textcoords='offset points', ha='center', fontsize=10)

# plt.tight_layout()
# plt.show()


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
w_theory = np.linspace(0.1, 1.0, 100)
zeta_boundary1 = K1 / (w_theory * K2)
zeta_boundary2 = K2 / (w_theory * K1)

# 有効な範囲のみプロット（ζ>1の場合も検証）
valid1 = (zeta_boundary1 >= 0.1) & (zeta_boundary1 <= 2.0)
valid2 = (zeta_boundary2 >= 0.1) & (zeta_boundary2 <= 2.0)

plt.plot(w_theory[valid1], zeta_boundary1[valid1], 'b--', linewidth=2, 
         label=f'理論境界: w = K₁/(ζ×K₂)')
# plt.plot(w_theory[valid2], zeta_boundary2[valid2], 'g--', linewidth=2, 
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
ax2 = axes[1]
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

# 3. 最初の絶滅時刻マップ
# ax3 = axes[2]
# # 最初の絶滅時刻を計算
# first_extinction_times = np.full_like(extinction_t1, np.nan)
# for i in range(len(extinction_t1)):
#     times = []
#     if not np.isnan(extinction_t1[i]):
#         times.append(extinction_t1[i])
#     if not np.isnan(extinction_t2[i]):
#         times.append(extinction_t2[i])
#     if times:
#         first_extinction_times[i] = min(times)

# first_extinction_mask = ~np.isnan(first_extinction_times)
# if np.any(first_extinction_mask):
#     first_times = first_extinction_times[first_extinction_mask]
#     vmin = np.percentile(first_times, 5)
#     vmax = np.percentile(first_times, 95)
    
#     scatter3 = ax3.scatter(w_vals[first_extinction_mask], zeta_vals[first_extinction_mask], 
#                           c=first_times, cmap='inferno', 
#                           alpha=0.8, s=50, vmin=vmin, vmax=vmax)
#     plt.colorbar(scatter3, ax=ax3, label='最初の絶滅時刻', shrink=0.8)
# ax3.set_xlabel('w (競争係数)')
# ax3.set_ylabel('ζ (減衰率)')
# ax3.set_title('最初の絶滅時刻マップ')
# ax3.grid(True, alpha=0.3)

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
# 7. 絶滅時刻の詳細分析と可視化
# --------------------------------------------------------------------------




zeta_range = np.linspace(0, 2.0, 50)
