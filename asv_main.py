#!/usr/bin/env python
# coding=utf-8

from asv_env import ASVEnv
import numpy as np
import time
from asv_agent import DDPG
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class RandAgent():
    def get_action_noise(self, s):
        return np.array([10, 10]) + (np.random.random(2) - 0.5) * 5

MAX_EPISODE = 1000000
MAX_DECAYEP = 3000
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
    # agent = RandAgent()
    if need_load:
        START_EPISODE = agent.load()
    else:
        START_EPISODE = 0

    summary_writer = agent.get_summary_writer()
    show_reward = 0
    for e in range(START_EPISODE, MAX_EPISODE):
        cur_state = env.reset()
        cum_reward = 0
        noise_decay_rate = max((MAX_DECAYEP - e) / MAX_DECAYEP, 0.1)
        agent.build_noise(0, 1 * noise_decay_rate)  # 根据给定的均值和decay的方差，初始化噪声发生器

        for step in range(MAX_STEP):

            action = agent.get_action_noise(cur_state)

            next_state, reward, done, info = env.step(action)

            # reward = float(reward / 10)

            agent.add_step(cur_state, action, reward, done, next_state)
            agent.learn_batch()

            # info = {
            #     "cur_state": list(cur_state), "action": list(action),
            #     "next_state": list(next_state), "reward": reward, "done": done
            # }
            info = {
                "ship": list(np.append(env.asv.position.data, env.asv.velocity.data)), "action": list(action),
                "aim": list(env.aim.position.data), "reward": reward, "done": done
            }
            # print(info, flush=True)

            cur_state = next_state
            cum_reward += reward
            show_reward += reward
            if agent.run_step % MAX_STEP == 0:
                summary_writer.add_scalar('cum_reward', show_reward, agent.run_step)
                show_reward = 0

            if RENDER:
                env.render()
                time.sleep(0.1)

            done = done or step == MAX_STEP - 1
            if done:
                print(f'episode: {e}, cum_reward: {cum_reward}, step_num:{step+1}', flush=True)
                # if cum_reward < 1:
                #     RENDER = True
                break
        # summary_writer.add_scalar('cum_reward', cum_reward, e)
        agent.save(e)  # 保存网络参数


if __name__ == '__main__':
    rl_loop()
