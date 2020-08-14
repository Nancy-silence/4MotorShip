#!/usr/bin/env python
# coding=utf-8

from asv_env import ASVEnv
import numpy as np
import pandas as pd
import time
from asv_agent import DDPG
import os
import signal

class GracefulExitException(Exception):
    @staticmethod
    def sigterm_handler(signum, frame):
        raise GracefulExitException()
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MAX_EPISODE = 1000000
# MAX_DECAYEP = 3000
MAX_STEP = 300

LR_A = 0.0005
LR_C = 0.001

def rl_loop(model_path=False):
    """
    @param:   

    model_path : 默认False表示全新的训练;需要加载模型则要传入模型路径 eg:'./model/linear.pth'
    """
    try:
        RENDER = False

        env = ASVEnv(target_trajectory='func_sin')
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space.high[0]

        agent = DDPG(s_dim, a_dim, a_bound, lr_a=LR_A, lr_c=LR_C, gamma=0.95, MAX_MEM=300000, MIN_MEM=1000, BATCH_SIZE=128)
        if model_path != False:
            START_EPISODE = agent.load(model_path)
        else:
            START_EPISODE = 0

        summary_writer = agent.get_summary_writer()

        reward_his = []
        best_ave_cumreward = -1000
        ten_episode_ave_reward = -6.3

        for e in range(START_EPISODE, MAX_EPISODE):
            cur_state = env.reset()
            cum_reward = 0
            # noise_decay_rate = max((MAX_DECAYEP - e) / MAX_DECAYEP, 0.05)
            # noise_decay_rate = - np.exp(ten_episode_ave_reward - 0.5) + 1
            noise_decay_rate = -0.147 * ten_episode_ave_reward + 0.0735
            agent.build_noise(0, noise_decay_rate)  # 根据给定的均值和decay的方差，初始化噪声发生器

            for step in range(MAX_STEP):

                action = agent.get_action_noise(cur_state)

                next_state, reward, done, info = env.step(action)

                agent.add_step(cur_state, action, reward, done, next_state)
                agent.learn_batch()

                info = {
                    "ship": list(np.append(env.asv.position.data, env.asv.velocity.data)), "action": list(action),
                    "aim": list(env.aim.position.data), "reward": reward, "done": done
                }
                # print(info, flush=True)

                cur_state = next_state
                cum_reward += reward

                if RENDER:
                    env.render()
                    time.sleep(0.1)

                if done or step == MAX_STEP - 1:
                    summary_writer.add_scalar('reward', cum_reward/(step+1), e+1)
                    print(f'episode: {e}, cum_reward: {cum_reward}, step_num:{step+1}', flush=True)
                    reward_his.append([e, cum_reward, step+1])
                    # if cum_reward > -10:
                    #     RENDER = True
                    break

            # 计算近期平均reward,为adaptive noise decay rate准备
            ten_episode_ave_reward = 0
            counter = 0
            for i in reward_his[-min(50, len(reward_his)):]:
                counter += 1
                ten_episode_ave_reward += i[1] / MAX_STEP
            ten_episode_ave_reward /= counter

            #模型保存

            # 覆盖保存每一局的模型
            agent.save(e, env.target_trajectory)
            # 保存最优模型
            signal = True
            cumreward_sum = 0
            for i in reward_his[-min(10, len(reward_his)):]:
                if i[2] != MAX_STEP:
                    signal = False
                    break
                else:
                    cumreward_sum += i[1]
            if signal and (cumreward_sum/ 10.0) > best_ave_cumreward:
                best_ave_cumreward = cumreward_sum / 10.0
                agent.save(e, env.target_trajectory + ' best_model')

    except (KeyboardInterrupt,GracefulExitException):
        reward_his = np.array(reward_his)
        data_save_exl(reward_his)

def data_save_exl(data_list):
    writer = pd.ExcelWriter('Reward.xlsx')
    data = pd.DataFrame({'episode':data_list[:,0],'reward':data_list[:,1],'step_num':data_list[:,2]})
    data.to_excel(writer,'sheet')
    writer.save()
    writer.close()

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)
    rl_loop()

