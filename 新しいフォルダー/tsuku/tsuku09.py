import gym
import numpy as np
import matplotlib.pyplot as plt

# 動画の描画関数の宣言
from matplotlib import animation

def display_frames_as_gif(frames):
    def animate(i):
        patch.set_data(frames[i])
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)
    anim.save('movie_cartpole.mp4')
    display(display_animation(anim, default_mode='loop'))

frames = []
env = gym.make('CartPole-v0')
env.reset()
for step in range(200):
    env.render()
    action = np.random.choice(2)
    obs, reward, done, _ = env.step(action)











#
