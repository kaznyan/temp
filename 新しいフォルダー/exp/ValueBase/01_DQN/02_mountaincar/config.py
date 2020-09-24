import torch

env_name = 'MountainCar-v0'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10

### 上手くいっていないときの1 episodeが長いもの（MountainCarなど）は10くらいまで小さく
### 上手くいっていないときの1 episodeが短いもの（CartPoleなど）は100くらいまで大きく
update_target = 10

replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
