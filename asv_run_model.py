#!/usr/bin/env python
# coding=utf-8

from asv_env import ASVEnv
import time
from asv_agent import DDPG
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MAX_EPISODE = 10000000
MAX_DECAYEP = 500
MAX_STEP = 300

LR_A = 0.0005
LR_C = 0.001


def rl_loop(need_load=True):
    RENDER = False

    env = ASVEnv(target_trajectory='func_sin')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    agent = DDPG(s_dim, a_dim, a_bound, lr_a=LR_A, lr_c=LR_C, MAX_MEM=10000, MIN_MEM=1000, BATCH_SIZE=128)
    if need_load:
        START_EPISODE = agent.load()
    else:
        START_EPISODE = 0

    for e in range(START_EPISODE, MAX_EPISODE):
        cur_state = env.reset()
        cum_reward = 0
        for step in range(MAX_STEP):
            action = agent.get_action(cur_state)[0]

            next_state, reward, done, info = env.step(action)

            # reward = float(reward / 10)

            # info = {
            #     "cur_state": list(cur_state), "action": list(action),
            #     "next_state": list(next_state), "reward": reward, "done": done
            # }
            info = {
                "ship": list(np.append(env.asv.position.data, env.asv.velocity.data)), "action": list(action),
                "aim": list(env.aim.position.data), "reward": reward, "done": done
            }
            print(info, flush=True)

            cur_state = next_state
            cum_reward += reward
            if RENDER:
                env.render()
                time.sleep(0.1)

            done = done or step == MAX_STEP - 1
            if done:
                print(f'episode: {e}, cum_reward: {cum_reward}', flush=True)
                if cum_reward < 1:
                    RENDER = True
                break

if __name__ == '__main__':
    rl_loop()
