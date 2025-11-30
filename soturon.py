import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# 定数と設定
# ============================================================================

# 日本語フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 
                                    'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# シミュレーション定数
EXTINCTION_THRESHOLD = 1e-3
COEXISTENCE_THRESHOLD = 1e-3
SOLVER_MAX_STEP = 1.0
SOLVER_RTOL = 1e-3
SOLVER_ATOL = 1e-3

# デフォルトパラメータ
DEFAULT_T_SPAN = [0, 100]
DEFAULT_T_EVAL_POINTS = 500
DEFAULT_INITIAL_CONDITIONS = [10, 10, 0, 0]
DEFAULT_R1, DEFAULT_R2 = 2.0, 2.0
DEFAULT_K1, DEFAULT_K2 = 100, 1000
DEFAULT_ZETA1, DEFAULT_ZETA2 = 0.8, 0.8

# ============================================================================
# コア関数: 微分方程式とイベント
# ============================================================================

def attention_dynamics(t, y, r1, r2, K1, K2, w12, w21, zeta1, zeta2):
    """
    アテンションダイナミクスの連立微分方程式
    
    Parameters:
    -----------
    t : float
        時間
    y : array-like
        状態ベクトル [a1, a2, b1, b2]
    r1, r2 : float
        成長率
    K1, K2 : float
        環境収容力
    w12, w21 : float
        競争係数
    zeta1, zeta2 : float
        減衰率
    
    Returns:
    --------
    list
        微分方程式の右辺 [da1/dt, da2/dt, db1/dt, db2/dt]
    """
    a1, a2, b1, b2 = y
    da1_dt = r1 * a1 * (1 - (b1 + w12 * a2) / K1)
    da2_dt = r2 * a2 * (1 - (b2 + w21 * a1) / K2)
    db1_dt = a1 - zeta1 * b1
    db2_dt = a2 - zeta2 * b2
    return [da1_dt, da2_dt, db1_dt, db2_dt]


def create_extinction_event(threshold=EXTINCTION_THRESHOLD):
    """絶滅イベント関数を作成"""
    def event_extinction_a1(t, y, *params):
        return y[0] - threshold
    event_extinction_a1.terminal = True
    event_extinction_a1.direction = -1
    
    def event_extinction_a2(t, y, *params):
        return y[1] - threshold
    event_extinction_a2.terminal = True
    event_extinction_a2.direction = -1
    
    return event_extinction_a1, event_extinction_a2


# グローバルイベント関数（後方互換性のため）
event_extinction_a1, event_extinction_a2 = create_extinction_event()


def calculate_K_ratio(w11, w12, w21, w22):
    """
    競争度行列の要素からK1/K2を計算
    
    K1/K2 = sqrt(w_21 * w_22 / (w_12 * w_11))
    """
    return np.sqrt((w21 * w22) / (w12 * w11))


# ============================================================================
# シミュレーション関数
# ============================================================================

def create_params(r1, r2, K1, K2, w12, w21, zeta1, zeta2):
    """パラメータ辞書を作成"""
    return {
        'r1': r1, 'r2': r2,
        'K1': K1, 'K2': K2,
        'w12': w12, 'w21': w21,
        'zeta1': zeta1, 'zeta2': zeta2
    }


def run_simulation(params, initial_conditions, t_span, t_eval=None, 
                   events=None, use_advanced_solver=False):
    """
    シミュレーションを実行
    
    Parameters:
    -----------
    params : dict
        パラメータ辞書
    initial_conditions : array-like
        初期条件 [a1(0), a2(0), b1(0), b2(0)]
    t_span : list
        時間範囲 [t_start, t_end]
    t_eval : array-like, optional
        評価する時間点
    events : list, optional
        イベント関数のリスト
    use_advanced_solver : bool
        Trueの場合、より厳密なソルバー設定を使用
    
    Returns:
    --------
    scipy.integrate.OdeResult
        シミュレーション結果
    """
    solver_kwargs = {
        'dense_output': True,
        't_eval': t_eval,
    }
    
    if use_advanced_solver:
        solver_kwargs.update({
            'method': 'Radau',
            'max_step': SOLVER_MAX_STEP,
            'rtol': SOLVER_RTOL,
            'atol': SOLVER_ATOL
        })
    
    if events is not None:
        solver_kwargs['events'] = events
    
    return solve_ivp(
        attention_dynamics,
        t_span,
        initial_conditions,
        args=tuple(params.values()),
        **solver_kwargs
    )


def check_coexistence(sol, threshold=COEXISTENCE_THRESHOLD):
    """
    共存判定を行う
    
    Parameters:
    -----------
    sol : scipy.integrate.OdeResult
        シミュレーション結果
    threshold : float
        共存判定の閾値
    
    Returns:
    --------
    tuple
        (is_coexistence, a1_final, a2_final)
    """
    if sol is None or not sol.success:
        return False, 0.0, 0.0
    
    a1_final = sol.y[0][-1]
    a2_final = sol.y[1][-1]
    is_coexistence = (a1_final > threshold) and (a2_final > threshold)
    return is_coexistence, a1_final, a2_final


def calculate_extinction_times(sol, threshold=EXTINCTION_THRESHOLD):
    """
    絶滅時刻を計算
    
    Parameters:
    -----------
    sol : scipy.integrate.OdeResult
        シミュレーション結果
    threshold : float
        絶滅判定の閾値
    
    Returns:
    --------
    tuple
        (extinction_time_a1, extinction_time_a2)
    """
    extinction_time_a1 = np.nan
    extinction_time_a2 = np.nan
    
    if sol is None:
        return extinction_time_a1, extinction_time_a2
    
    # イベントから絶滅時刻を取得
    if hasattr(sol, 't_events') and sol.t_events is not None and len(sol.t_events) >= 2:
        if len(sol.t_events[0]) > 0:
            extinction_time_a1 = sol.t_events[0][0]
        if len(sol.t_events[1]) > 0:
            extinction_time_a2 = sol.t_events[1][0]
    else:
        # イベントがない場合、時系列データから計算
        try:
            a1_below = np.where(sol.y[0] <= threshold)[0]
            a2_below = np.where(sol.y[1] <= threshold)[0]
            extinction_time_a1 = sol.t[a1_below[0]] if a1_below.size > 0 else np.nan
            extinction_time_a2 = sol.t[a2_below[0]] if a2_below.size > 0 else np.nan
        except (IndexError, AttributeError):
            pass
    
    return extinction_time_a1, extinction_time_a2


def check_coexistence_stability(w12, w21, zeta1, zeta2, K1, K2, r1, r2, 
                               initial_conditions, t_span, t_eval):
    """
    数値シミュレーションで共存の安定性をチェック
    
    Returns:
    --------
    tuple
        (is_coexistence, a1_final, a2_final, sol)
    """
    try:
        params = create_params(r1, r2, K1, K2, w12, w21, zeta1, zeta2)
        sol = run_simulation(params, initial_conditions, t_span, t_eval, 
                            #events=[event_extinction_a1, event_extinction_a2],
                           use_advanced_solver=True)
        is_coexistence, a1_final, a2_final = check_coexistence(sol)
        return is_coexistence, a1_final, a2_final, sol
    except Exception as e:
        print(f"シミュレーションエラー: w12={w12}, w21={w21}, zeta1={zeta1}, zeta2={zeta2}, エラー: {e}")
        return False, 0, 0, None


# ============================================================================
# パラメータスイープ関数
# ============================================================================

def parameter_sweep_analysis(w_range, zeta_range, K1, K2, r1, r2, 
                            initial_conditions, t_span, t_eval, 
                            progress_interval=50):
    """
    パラメータスイープで共存領域を効率的に探索（絶滅時刻も記録）
    
    Returns:
    --------
    tuple
        (w_values, zeta_values, coexistence_results, a1_final_values, 
         a2_final_values, extinction_times_a1, extinction_times_a2)
    """
    print("パラメータスイープを開始...")
    
    results = {
        'w_values': [],
        'zeta_values': [],
        'coexistence_results': [],
        'a1_final_values': [],
        'a2_final_values': [],
        'extinction_times_a1': [],
        'extinction_times_a2': []
    }
    
    total_combinations = len(w_range) * len(zeta_range)
    current = 0
    
    for w in w_range:
        for zeta in zeta_range:
            current += 1
            if current % progress_interval == 0:
                print(f"進行状況: {current}/{total_combinations} "
                      f"({current/total_combinations*100:.1f}%) - "
                      f"w={w:.3f}, zeta={zeta:.3f}")
            
            try:
                is_coexistence, a1_final, a2_final, sol = check_coexistence_stability(
                    w, w, zeta, zeta, K1, K2, r1, r2, 
                    initial_conditions, t_span, t_eval
                )
                
                extinction_time_a1, extinction_time_a2 = calculate_extinction_times(sol)
                
                results['w_values'].append(w)
                results['zeta_values'].append(zeta)
                results['coexistence_results'].append(is_coexistence)
                results['a1_final_values'].append(a1_final)
                results['a2_final_values'].append(a2_final)
                results['extinction_times_a1'].append(extinction_time_a1)
                results['extinction_times_a2'].append(extinction_time_a2)
                
            except Exception as e:
                print(f"警告: w={w:.3f}, zeta={zeta:.3f} でエラー: {e}")
                results['w_values'].append(w)
                results['zeta_values'].append(zeta)
                results['coexistence_results'].append(False)
                results['a1_final_values'].append(0)
                results['a2_final_values'].append(0)
                results['extinction_times_a1'].append(np.nan)
                results['extinction_times_a2'].append(np.nan)
    
    return (np.array(results['w_values']), np.array(results['zeta_values']), 
            np.array(results['coexistence_results']), 
            np.array(results['a1_final_values']), 
            np.array(results['a2_final_values']), 
            np.array(results['extinction_times_a1']), 
            np.array(results['extinction_times_a2']))


def sweep_K_ratio_and_w(K_ratio_range, w_range, K2_fixed, zeta1, zeta2, 
                        r1, r2, initial_conditions, t_span, t_eval,
                        progress_interval=500):
    """
    K1/K2比とwの範囲でパラメータスイープを実行
    
    Returns:
    --------
    tuple
        (K_ratio_values, w_values, coexistence_results, a1_final_map, a2_final_map)
    """
    results = {
        'K_ratio_values': [],
        'w_values': [],
        'coexistence_results': [],
        'a1_final_map': [],
        'a2_final_map': []
    }
    
    total_combinations = len(K_ratio_range) * len(w_range)
    current = 0
    
    for K_ratio in K_ratio_range:
        for w_val in w_range:
            current += 1
            if current % progress_interval == 0:
                print(f"進行状況: {current}/{total_combinations} "
                      f"({current/total_combinations*100:.1f}%)")
            
            K1_val = K_ratio * K2_fixed
            
            try:
                is_coexistence, a1_final, a2_final, sol = check_coexistence_stability(
                    w_val, w_val, zeta1, zeta2, K1_val, K2_fixed, 
                    r1, r2, initial_conditions, t_span, t_eval
                )
                
                results['K_ratio_values'].append(K_ratio)
                results['w_values'].append(w_val)
                results['coexistence_results'].append(is_coexistence)
                results['a1_final_map'].append(a1_final)
                results['a2_final_map'].append(a2_final)
                
            except Exception as e:
                results['K_ratio_values'].append(K_ratio)
                results['w_values'].append(w_val)
                results['coexistence_results'].append(False)
                results['a1_final_map'].append(0)
                results['a2_final_map'].append(0)
    
    return (np.array(results['K_ratio_values']), 
            np.array(results['w_values']),
            np.array(results['coexistence_results']),
            np.array(results['a1_final_map']),
            np.array(results['a2_final_map']))


# ============================================================================
# 可視化関数
# ============================================================================

def plot_phase_portrait(w12, w21, zeta1, zeta2, K1, K2, r1, r2):
    """
    相平面上にベクトル場とヌルクラインを描画する
    （簡易化のためbは平衡状態 b = a/zeta を仮定して2次元に落とし込む）
    """
    x_range = np.linspace(0, K1*1.5, 20)
    y_range = np.linspace(0, K2*1.5, 20)
    A1, A2 = np.meshgrid(x_range, y_range)
    
    # ベクトル場の計算（bは準定常状態 b=a/zeta を仮定）
    B1 = A1 / zeta1
    B2 = A2 / zeta2
    U = r1 * A1 * (1 - (B1 + w12 * A2) / K1)
    V = r2 * A2 * (1 - (B2 + w21 * A1) / K2)
    
    plt.figure(figsize=(8, 8))
    plt.streamplot(A1, A2, U, V, color='gray', linewidth=1, density=1.5)
    
    # ヌルクラインの描画
    nullcline1_a2 = (K1 - x_range/zeta1) / w12
    nullcline2_a2 = zeta2 * (K2 - w21 * x_range)
    
    plt.plot(x_range, nullcline1_a2, 'b--', label=r'$\dot{a}_1=0$ Nullcline', linewidth=2)
    plt.plot(x_range, nullcline2_a2, 'r--', label=r'$\dot{a}_2=0$ Nullcline', linewidth=2)
    
    plt.xlim(0, K1*1.2)
    plt.ylim(0, K2*1.2)
    plt.xlabel('種1 注目度 $a_1$')
    plt.ylabel('種2 注目度 $a_2$')
    plt.title(f'相平面図 (w12={w12}, w21={w21})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_w_comparison(results):
    """w値別の注目度変化比較図を描画"""
    plt.figure(figsize=(12, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))

    # 種1
    plt.subplot(1, 2, 1)
    for i, (w, sol, _, _, is_coexistence, a1_final, a2_final, _, _) in enumerate(results):
        linestyle = '-' if is_coexistence else '--'
        linewidth = 3 if is_coexistence else 2
        plt.plot(
            sol.t,
            sol.y[0],
            label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}',
            color=colors[i],
            linewidth=linewidth,
            linestyle=linestyle,
        )
    plt.title('種1の注目度変化（w値別）')
    plt.xlabel('時間 t')
    plt.ylabel('注目度 $a_1(t)$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 種2
    plt.subplot(1, 2, 2)
    for i, (w, sol, _, _, is_coexistence, a1_final, a2_final, _, _) in enumerate(results):
        linestyle = '-' if is_coexistence else '--'
        linewidth = 3 if is_coexistence else 2
        plt.plot(
            sol.t,
            sol.y[1],
            label=f'w={w:.2f} {"共存" if is_coexistence else "排除"}',
            color=colors[i],
            linewidth=linewidth,
            linestyle=linestyle,
        )
    plt.title('種2の注目度変化（w値別）')
    plt.xlabel('時間 t')
    plt.ylabel('注目度 $a_2(t)$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('w値別の注目度変化比較', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_w_detail(results):
    """w値別の詳細図を描画"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, (w, sol, _, _, is_coexistence, _, _, _, _) in enumerate(results):
        ax = axes[i]
        ax.plot(sol.t, sol.y[0], color='blue', linewidth=2, label='種1')
        ax.plot(sol.t, sol.y[1], color='red', linewidth=2, label='種2')
        ax.set_title('競争が比較的小さい' if w < 0.6 else '競争が比較的大きい')
        ax.set_xlabel('時間 t')
        ax.set_ylabel('注目度')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_w_zeta_coexistence(w_vals, zeta_vals, coexistence_mask, 
                            extinction_t1=None, extinction_t2=None):
    """wとzetaの共存範囲マップを描画"""
    plt.figure(figsize=(8, 6))

    # 理論的境界線
    w_theory = np.linspace(0.1, 1.0, 100)
    zeta_boundary1 = DEFAULT_K1 / (w_theory * DEFAULT_K2)
    valid1 = (zeta_boundary1 >= 0.1) & (zeta_boundary1 <= 2.0)
    plt.plot(
        w_theory[valid1],
        zeta_boundary1[valid1],
        'b--',
        linewidth=2,
        label='理論境界: w = K₁/(ζ×K₂)',
    )

    # 数値結果
    plt.scatter(
        w_vals[coexistence_mask],
        zeta_vals[coexistence_mask],
        color='green',
        alpha=0.6,
        s=40,
        label='数値: 共存可能',
    )
    plt.scatter(
        w_vals[~coexistence_mask],
        zeta_vals[~coexistence_mask],
        color='red',
        alpha=0.6,
        s=40,
        label='数値: 共存不可能',
    )

    plt.xlabel('w (競争係数)')
    plt.ylabel('ζ (減衰率)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 絶滅時刻マップ
    if extinction_t1 is not None:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        extinction_mask_a1 = ~np.isnan(extinction_t1)
        if np.any(extinction_mask_a1):
            extinction_times_a1 = extinction_t1[extinction_mask_a1]
            vmin = np.percentile(extinction_times_a1, 5)
            vmax = np.percentile(extinction_times_a1, 95)
            scatter1 = ax1.scatter(
                w_vals[extinction_mask_a1],
                zeta_vals[extinction_mask_a1],
                c=extinction_times_a1,
                cmap='plasma',
                alpha=0.8,
                s=50,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(scatter1, ax=ax1, label='絶滅時刻', shrink=0.8)
        ax1.set_xlabel('w (競争係数)')
        ax1.set_ylabel('ζ (減衰率)')
        ax1.set_title('種1の絶滅時刻マップ')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_K_ratio_coexistence(K_ratio_values, w_values, coexistence_results, 
                             a1_final_map, a2_final_map, zeta1, zeta2):
    """K1/K2比とwの共存範囲マップを描画"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 共存判定マップ（2値）
    ax1 = axes[0]
    ax1.scatter(K_ratio_values[coexistence_results], 
               w_values[coexistence_results], 
               c='green', alpha=0.6, s=20, label='共存可能')
    ax1.scatter(K_ratio_values[~coexistence_results], 
               w_values[~coexistence_results], 
               c='red', alpha=0.6, s=20, label='共存不可能')
    ax1.set_xlabel('K₁/K₂ 比', fontsize=12)
    ax1.set_ylabel('w (競争係数)', fontsize=12)
    ax1.set_title('K₁/K₂比とwによる共存範囲マップ', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5.0)
    ax1.set_ylim(0, 1.0)
    
    # 2. 最終注目度の合計による強度マップ
    ax2 = axes[1]
    total_attention = a1_final_map + a2_final_map
    coexistence_mask = coexistence_results & (total_attention > 1e-3)
    if np.any(coexistence_mask):
        scatter3 = ax2.scatter(K_ratio_values[coexistence_mask], 
                              w_values[coexistence_mask], 
                              c=total_attention[coexistence_mask], 
                              cmap='viridis', alpha=0.8, s=20, 
                              vmin=np.percentile(total_attention[coexistence_mask], 5),
                              vmax=np.percentile(total_attention[coexistence_mask], 95))
        plt.colorbar(scatter3, ax=ax2, label='最終注目度の合計 (a₁ + a₂)', shrink=0.8)
    ax2.scatter(K_ratio_values[~coexistence_results], 
               w_values[~coexistence_results], 
               c='lightgray', alpha=0.3, s=20)
    ax2.set_xlabel('K₁/K₂ 比', fontsize=12)
    ax2.set_ylabel('w (競争係数)', fontsize=12)
    ax2.set_title('共存領域の安定性マップ（最終注目度の合計）', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5.0)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

    # 3. 理論境界との比較
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(K_ratio_values[coexistence_results], 
              w_values[coexistence_results], 
              c='green', alpha=0.6, s=20, label='数値: 共存可能')
    ax.scatter(K_ratio_values[~coexistence_results], 
              w_values[~coexistence_results], 
              c='red', alpha=0.6, s=20, label='数値: 共存不可能')
    
    # 理論的境界線
    K_ratio_theory = np.linspace(0.1, 5.0, 100)
    w_boundary1 = K_ratio_theory / zeta1
    w_boundary2 = 1.0 / (zeta2 * K_ratio_theory)
    w_boundary = np.minimum(w_boundary1, w_boundary2)
    valid = w_boundary <= 1.0
    ax.plot(K_ratio_theory[valid], w_boundary[valid], 'b--', linewidth=2, 
            label=f'理論境界: w = min((K₁/K₂)/ζ, 1/(ζ·K₁/K₂))')
    
    ax.set_xlabel('K₁/K₂ 比', fontsize=12)
    ax.set_ylabel('w (競争係数)', fontsize=12)
    ax.set_title('K₁/K₂比とwによる共存範囲（理論境界との比較）', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.show()


def plot_K1_variation(K1_values, K2_fixed, a1_final_values, a2_final_values):
    """K1変化時の共存状態可視化"""
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    K_ratio = K1_values / K2_fixed
    ax.plot(K_ratio, a1_final_values, 'b-', linewidth=2, label='種1 ($a_1$)')
    ax.plot(K_ratio, a2_final_values, 'r-', linewidth=2, label='種2 ($a_2$)')
    ax.axhline(y=EXTINCTION_THRESHOLD, color='gray', linestyle='--', 
              alpha=0.5, label='絶滅閾値')
    ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.5, label='K₁=K₂')
    ax.set_xlabel('K₁/K₂ 比')
    ax.set_ylabel('最終注目度')
    ax.set_title('K₁/K₂比と最終注目度の関係')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 統計分析関数
# ============================================================================

def print_extinction_statistics(extinction_t1, extinction_t2):
    """絶滅時刻の統計を表示"""
    print("\n=== パラメータスイープ絶滅時刻分析 ===")

    extinction_occurred = ~(np.isnan(extinction_t1) & np.isnan(extinction_t2))
    extinction_rate = np.mean(extinction_occurred)
    print(
        f"絶滅発生率: {extinction_rate*100:.1f}% "
        f"({np.sum(extinction_occurred)}/{len(extinction_occurred)} ケース)"
    )

    if np.any(extinction_occurred):
        # 種1の絶滅統計
        a1_extinct = ~np.isnan(extinction_t1)
        if np.any(a1_extinct):
            print("\n種1絶滅統計:")
            print(f"  発生率: {np.mean(a1_extinct)*100:.1f}%")
            print(f"  平均絶滅時刻: {np.nanmean(extinction_t1):.2f}")
            print(f"  絶滅時刻の標準偏差: {np.nanstd(extinction_t1):.2f}")

        # 種2の絶滅統計
        a2_extinct = ~np.isnan(extinction_t2)
        if np.any(a2_extinct):
            print("\n種2絶滅統計:")
            print(f"  発生率: {np.mean(a2_extinct)*100:.1f}%")
            print(f"  平均絶滅時刻: {np.nanmean(extinction_t2):.2f}")
            print(f"  絶滅時刻の標準偏差: {np.nanstd(extinction_t2):.2f}")
    
    
def print_sweep_statistics(w_vals, coexistence_mask, a1_finals, a2_finals):
    """パラメータスイープ結果の統計を表示"""
    print("\n=== パラメータスイープ結果 ===")
    print(f"総テスト数: {len(w_vals)}")
    print(
        f"共存可能な組み合わせ: {np.sum(coexistence_mask)} "
        f"({np.mean(coexistence_mask)*100:.1f}%)"
    )
    print(
        f"共存不可能な組み合わせ: {np.sum(~coexistence_mask)} "
        f"({np.mean(~coexistence_mask)*100:.1f}%)"
    )

    # 最も安定した共存パラメータを特定
    if np.any(coexistence_mask):
        stable_indices = coexistence_mask & (a1_finals > 5) & (a2_finals > 5)
        if np.any(stable_indices):
            best_idx = np.argmax(a1_finals[stable_indices] + a2_finals[stable_indices])
            best_w = w_vals[stable_indices][best_idx]
            print("\n最も安定した共存パラメータ:")
            print(f"w = {best_w:.3f}")
            print(
                f"最終注目度: a1 = {a1_finals[stable_indices][best_idx]:.3f}, "
                f"a2 = {a2_finals[stable_indices][best_idx]:.3f}"
            )


# ============================================================================
# メイン処理
# ============================================================================

def main():
    """メイン処理"""
    # 共通設定
    t_span = DEFAULT_T_SPAN
    t_eval = np.linspace(t_span[0], t_span[1], DEFAULT_T_EVAL_POINTS)
    initial_conditions = DEFAULT_INITIAL_CONDITIONS
    r1, r2 = DEFAULT_R1, DEFAULT_R2
    K1, K2 = DEFAULT_K1, DEFAULT_K2
    zeta1, zeta2 = DEFAULT_ZETA1, DEFAULT_ZETA2
    
    # ========================================================================
    # シナリオ1: 共存ケース
    # ========================================================================
    params_coexistence = create_params(r1, r2, K1, K2, 1.0, 1.0, zeta1, zeta2)
    sol_coexistence = run_simulation(params_coexistence, initial_conditions, 
                                    t_span, t_eval)
    
    # ========================================================================
    # シナリオ2: 競合的排除ケース
    # ========================================================================
    params_exclusion = create_params(r1, r2, K1, K2, 0.97, 0.95, zeta1, zeta2)
    sol_exclusion = run_simulation(params_exclusion, initial_conditions, 
                                  t_span, t_eval)
    
    # ========================================================================
    # w値別のシミュレーション
    # ========================================================================
    w_list = [0.2, 0.95]
    results = []
    w11, w22 = 1.0, 1.0
    
    for w in w_list:
        w12 = w21 = w
        K_ratio = calculate_K_ratio(w11, w12, w21, w22)
        params = create_params(r1, r2, K1, K2, w12, w21, zeta1, zeta2)
        sol = run_simulation(params, initial_conditions, t_span, t_eval)
        
        is_coexistence, a1_final, a2_final = check_coexistence(sol)
        extinction_time_a1, extinction_time_a2 = calculate_extinction_times(sol)
        
        results.append((w, sol, 0, K_ratio, is_coexistence, a1_final, a2_final, 
                       extinction_time_a1, extinction_time_a2))
    
    # 可視化
    plot_w_comparison(results)
    plot_w_detail(results)
    
    # ========================================================================
    # wとzetaのパラメータスイープ
    # ========================================================================
    w_sweep = np.linspace(0.1, 1.0, 40)
    zeta_sweep = np.linspace(0.1, 2.0, 40)
    
    w_vals, zeta_vals, coexistence_mask, a1_finals, a2_finals, \
        extinction_t1, extinction_t2 = parameter_sweep_analysis(
            w_sweep, zeta_sweep, K1, K2, r1, r2, 
            initial_conditions, t_span, t_eval
        )
    
    plot_w_zeta_coexistence(w_vals, zeta_vals, coexistence_mask, 
                           extinction_t1, extinction_t2)
    print_extinction_statistics(extinction_t1, extinction_t2)
    print_sweep_statistics(w_vals, coexistence_mask, a1_finals, a2_finals)
    
    # ========================================================================
    # 相平面図
    # ========================================================================
    plot_phase_portrait(1.0, 1.0, zeta1, zeta2, K1, K2, r1, r2)
    
    # ========================================================================
    # K1変化時の分析
    # ========================================================================
    print("\n=== K2を固定し、K1を変化させた時の共存状態分析 ===")
    K2_fixed = 1000
    K1_range = np.linspace(50, 2000, 50)
    K1_values = []
    a1_final_values = []
    a2_final_values = []
    coexistence_flags = []

    w12_test = w21_test = 1.0

    print(f"K2 = {K2_fixed} に固定")
    print(f"K1の範囲: {K1_range[0]:.1f} ～ {K1_range[-1]:.1f}")
    print("シミュレーション実行中...")

    for K1_val in K1_range:
        params = create_params(r1, r2, K1_val, K2_fixed, w12_test, w21_test,
                               zeta1, zeta2)
        try:
            sol = run_simulation(
                params,
                initial_conditions,
                t_span,
                t_eval,
                events=[event_extinction_a1, event_extinction_a2],
            )
            is_coexistence, a1_final, a2_final = check_coexistence(sol)

            K1_values.append(K1_val)
            a1_final_values.append(a1_final)
            a2_final_values.append(a2_final)
            coexistence_flags.append(is_coexistence)
        except Exception as e:
            print(f"エラー: K1={K1_val:.1f} でシミュレーション失敗: {e}")
            continue

    K1_values = np.array(K1_values)
    a1_final_values = np.array(a1_final_values)
    a2_final_values = np.array(a2_final_values)
    coexistence_flags = np.array(coexistence_flags)

    print("\n共存可能なK1値の範囲:")
    if np.any(coexistence_flags):
        coexisting_K1 = K1_values[coexistence_flags]
        print(f"  最小K1: {np.min(coexisting_K1):.1f}")
        print(f"  最大K1: {np.max(coexisting_K1):.1f}")
        print(f"  共存可能な割合: {np.mean(coexistence_flags)*100:.1f}%")
    else:
        print("  共存可能なK1値は見つかりませんでした")

    plot_K1_variation(K1_values, K2_fixed, a1_final_values, a2_final_values)
    
    # ========================================================================
    # K1/K2比とwの共存範囲マップ
    # ========================================================================
    print("\n=== K1/K2比とwの共存範囲分析 ===")
    K_ratio_range = np.linspace(0.1, 5.0, 50)
    w_range_coexistence = np.linspace(0.1, 1.0, 50)
    K2_fixed_coexistence = 1000
    
    print(f"K2 = {K2_fixed_coexistence} に固定")
    print(f"K1/K2比の範囲: {K_ratio_range[0]:.2f} ～ {K_ratio_range[-1]:.2f}")
    print(f"wの範囲: {w_range_coexistence[0]:.2f} ～ {w_range_coexistence[-1]:.2f}")
    print("シミュレーション実行中...")
    
    K_ratio_values, w_values_coexistence, coexistence_results_map, \
        a1_final_map, a2_final_map = sweep_K_ratio_and_w(
            K_ratio_range, w_range_coexistence, K2_fixed_coexistence,
            zeta1, zeta2, r1, r2, initial_conditions, t_span, t_eval
        )
    
    print(f"\n共存可能な組み合わせ: {np.sum(coexistence_results_map)}/"
          f"{len(coexistence_results_map)} "
          f"({np.mean(coexistence_results_map)*100:.1f}%)")
    
    plot_K_ratio_coexistence(K_ratio_values, w_values_coexistence, 
                            coexistence_results_map, a1_final_map, a2_final_map,
                            zeta1, zeta2)


if __name__ == "__main__":
    main()
