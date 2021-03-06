import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 5000
goal_score = 200
log_interval = 10
update_target = 500
replay_memory_capacity = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
