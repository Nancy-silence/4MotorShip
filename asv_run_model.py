#!/usr/bin/env python
# coding=utf-8

from asv_env import ASVEnv
import time
from asv_agent import DDPG
import numpy as np
import os

MAX_EPISODE = 10000000
MAX_STEP = 300

def rl_loop(model_path=False, render=True):
    RENDER = render

    env = ASVEnv(target_trajectory='func_sin')  # 加入测量误差用 measure_bias=True
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    agent = DDPG(s_dim, a_dim, a_bound, train = False)
    if model_path != False:
        START_EPISODE = agent.load(model_path)
    else:
        START_EPISODE = 0

    for e in range(START_EPISODE, START_EPISODE+10):
        cur_state = env.reset()
        cum_reward = 0
        for step in range(MAX_STEP):
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(action)

            info = {
                "cur_state": list(cur_state), "action": list(action),
                "next_state": list(next_state), "reward": reward, "done": done
            }
            # info = {
            #     "ship": list(np.append(env.asv.position.data, env.asv.velocity.data)), "action": list(action),
            #     "aim": list(env.aim.position.data), "reward": reward, "done": done
            # }
            # print(info, flush=True)

            cur_state = next_state
            cum_reward += reward
            if RENDER:
                env.render()
                time.sleep(0.1)

            done = done or step == MAX_STEP - 1
            if done:
                print(f'episode: {e}, cum_reward: {cum_reward}', flush=True)
                break
        env.data_save_exl()

if __name__ == '__main__':
    rl_loop('./model/func_sin best_model.pth')
