import numpy as np

"""
甲斐コメント
方策勾配法と言っているが厳密ではないように思える。
あえてだと思うが、報酬を定義せずに議論し、
「なんか迷路やらせてみて登場回数が大きい行動」＝良い行動　としている。
あまり理解が進まなかった。
報酬をちゃんと定義するのは7から。
"""

# --- 方策パラメータthetaを行動方策piに変換する ---
def simple_convert_into_pi_from_theta(theta):
    m, n = theta.shape  # thetaの行列サイズを取得
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 割合の計算
    pi = np.nan_to_num(pi)  # nanを0に変換
    return pi
def softmax_convert_into_pi_from_theta(theta):
    m, n = theta.shape  # thetaの行列サイズを取得
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 割合の計算
    pi = np.nan_to_num(pi)  # nanを0に変換
    return pi

### --- 1step移動後の状態 s を求める ---
def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    # pi[s, :]の確率に従って、directionを選択
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    elif next_direction == "left":
        action = 3
        s_next = s - 1
    return [action, s_next]

### --- エージェントの移動 ---
# 迷路を解く関数の定義、状態と行動の履歴を出力
def goal_maze_ret_s_a(pi):
    s = 0 ###スタート地点
    s_a_history = [[0, np.nan]] ### エージェントの移動を記録するリスト
    for i in range(100):
        action, next_s = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action ### 現在の状態の行動 a
        s_a_history.append([next_s, np.nan]) ### 次の状態 s'
        if next_s == 8:
            break
        s = next_s
    return s_a_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1  # 学習率
    T = len(s_a_history) - 1  # ゴールまでの総ステップ数

    m, n = theta.shape  # thetaの行列サイズを取得
    delta_theta = theta.copy()  # Δthetaの元を作成、ポインタ参照なので、delta_theta = thetaはダメ

    # delta_theta を要素ごとに求める
    for i in range(m):
        for j in range(n):
            if not (np.isnan(theta[i, j])):
                sa_i  = [sa for sa in s_a_history if sa[0] == i]    ### 履歴のうち状態iであったもの
                sa_ij = [sa for sa in s_a_history if sa == [i, j]] ### 履歴のうち状態iで行動jをしたもの
                n_i  = len(sa_i)  # 状態iで行動した総回数
                n_ij = len(sa_ij) # 状態iで行動jをとった回数
                delta_theta[i, j] = (n_ij + pi[i, j] * n_i) / T

    new_theta = theta + eta * delta_theta
    return new_theta


### --- 本編 ---

# 初期の方策を決定するパラメータ theta_0 を設定
# 行は状態0～7、列は移動方向で上、右、下、左
theta = np.array([[np.nan, 1, 1, np.nan], # s0
                  [np.nan, 1, np.nan, 1], # s1
                  [np.nan, np.nan, 1, 1], # s2
                  [1, 1, 1, np.nan], # s3
                  [np.nan, np.nan, 1, 1], # s4
                  [1, np.nan, np.nan, np.nan], # s5
                  [1, np.nan, np.nan, np.nan], # s6
                  [1, 1, np.nan, np.nan], # s7、※s8はゴールなので、方策はなし
                  ])

# 初期の方策pi_0を求める
pi = softmax_convert_into_pi_from_theta(theta)

for i in range(100000):
    # state_history = goal_maze(pi_0)
    s_a_history = goal_maze_ret_s_a(pi)

    theta = update_theta(theta, pi, s_a_history)

    pi = softmax_convert_into_pi_from_theta(theta)
    # print(pi)
    # print("ステップ数: " + str(len(s_a_history) - 1))

print(pi)
print("ステップ数: " + str(len(s_a_history) - 1))




#
