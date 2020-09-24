import numpy as np

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
        a = 0
        next_s = s - 3
    elif next_direction == "right":
        a = 1
        next_s = s + 1
    elif next_direction == "down":
        a = 2
        next_s = s + 3
    elif next_direction == "left":
        a = 3
        next_s = s - 1
    return a, next_s

def get_action_and_next_s_Q(s, Q, epsilon):
    ### ε-greedy法を実装
    direction = [0, 1, 2, 3] ### 上、右、下、左
    if np.random.rand() < epsilon: # ランダム
        while True:
            a = np.random.choice(direction)
            if not np.isnan(Q[s, a]):
                break
    else: # Qの最大値の行動を採用する
        a = direction[np.nanargmax(Q[s, :])]
    if a == 0:
        next_s = s - 3
    elif a == 1:
        next_s = s + 1
    elif a == 2:
        next_s = s + 3
    elif a == 3:
        next_s = s - 1
    return a, next_s


### --- エージェントの移動 ---
# 迷路を解く関数の定義、状態と行動の履歴を出力
def goal_maze_ret_s_a(pi):
    s = 0 ###スタート地点
    s_a_history = [[0, np.nan]] ### エージェントの移動を記録するリスト
    for i in range(100):
        a, next_s = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = a ### 現在の状態の行動 a
        s_a_history.append([next_s, np.nan]) ### 次の状態 s'
        if next_s == 8:
            break
        else:
            s = next_s
    return s_a_history

# Sarsaで迷路を解く関数の定義、状態と行動の履歴および更新したQを出力
def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma):
    s = 0 # スタート地点
    s_a_history = [[0, np.nan]] # エージェントの移動を記録するリスト
    for i in range(100):
        a, next_s = get_action_and_next_s_Q(s, Q, epsilon)
        s_a_history[-1][1] = a ### 現在の状態の行動 a
        s_a_history.append([next_s, np.nan]) ### 次の状態 s'
        if next_s == 8:
            r = 1
            next_a = np.nan
        else:
            r = 0
            ### 実際には行動しないが、次の行動 next_a を求める
            next_a, _ = get_action_and_next_s_Q(next_s, Q, epsilon)

        ### 行動価値関数を更新
        # Q = Sarsa(s, a, r, next_s, next_a, Q, eta, gamma)
        Q = Q_learning(s, a, r, next_s, Q, eta, gamma)

        if next_s == 8:
            break
        else:
            s = next_s
    return s_a_history, Q

### --- Q または theta の更新
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

def Sarsa(s, a, r, next_s, next_a, Q, eta, gamma):
    if next_s == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[next_s, next_a] - Q[s, a])
    return Q

def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])
    return Q

### --- 本編 ---

### 今回は Q を初期設定するためにのみ使う
### 行は状態0～7、列は移動方向で上、右、下、左
theta = np.array([[np.nan, 1, 1, np.nan], # s0
                  [np.nan, 1, np.nan, 1], # s1
                  [np.nan, np.nan, 1, 1], # s2
                  [1, 1, 1, np.nan], # s3
                  [np.nan, np.nan, 1, 1], # s4
                  [1, np.nan, np.nan, np.nan], # s5
                  [1, np.nan, np.nan, np.nan], # s6
                  [1, 1, np.nan, np.nan], # s7、※s8はゴールなので、方策はなし
                  ])
### pi_0 を決めるのではなく行動状態関数 Q(s, a) を決める
Q = np.random.rand(8, 4) * theta
### pi_0 は使わないが一応定義しておく
pi = simple_convert_into_pi_from_theta(theta)

# ### --- 方策勾配法 ---
# for i in range(1000):
#     s_a_history = goal_maze_ret_s_a(pi)
#
#     theta = update_theta(theta, pi, s_a_history)
#
#     pi = softmax_convert_into_pi_from_theta(theta)
#     # print(pi)
#     # print("ステップ数: " + str(len(s_a_history) - 1))

# --- Sarsa --
eta = 0.1 # 学習率
gamma = 0.9 # 時間割引率
epsilon = 0.5 # ε-greedy法の初期値
v = np.nanmax(Q, axis=1) # 状態ごとに価値の最大値を求める

for i in range(10000):
    epsilon = epsilon / 2
    ### Sarsaで迷路を解き、移動した履歴と更新したQを求める
    s_a_history, Q = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma)

    ### 状態価値の変化
    v = np.nanmax(Q, axis=1)  # 状態ごとに価値の最大値を求める

print(Q)
print("ステップ数: " + str(len(s_a_history) - 1))




#
