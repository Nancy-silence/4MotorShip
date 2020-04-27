## 4MotorShip

基于DDPG的4电机ASV轨迹追踪

### 环境配置

- Linux/Win  `python 3.6`
- 安装`pytorch 1.4.0`
- 需要安装的相关依赖包：numpy、matplotlib、tensorboard、gym

注意：
- 在Linux下运行需设置`c_env`文件夹下`step.py`文件中调用的动态链接库为`Sim.so`; Win下运行则设置为`Sim.dll`
```
sim = CDLL('./Sim.so')    # Win下运行设置为.dll
```
- 使用`Anaconda3`进行包管理，建议使用conda新建环境配置环境
### 下载项目

```
git clone https://github.com/Nancy-silence/4MotorShip.git
```

### 使用已训练好的模型 `asv_run_model.py`

1、选择要使用的模型（已训练好的模型在`model`文件夹下），在`asv_run_model.py`文件中将模型路径传入`rl_loop()`中，例如：

```
rl_loop('./model/func_sin.pth')
```

2、选择是否开启追踪可视化

设置`rl_loop()`的`render`参数，默认开启可视化，如需关闭设置为False

```
rl_loop('./model/func_sin.pth',False)
```

3、选择目标轨迹类型，修改`asv_run_model.py`中的target_trajectory（直线：linear；曲线：func_sin）

```
env = ASVEnv(target_trajectory='func_sin')
```

4、运行`asv_run_model.py`

控制台输出局次、本局累计奖励信息

figure显示实时追踪过程（黄色为追踪目标运动，蓝色为船的追踪运动）

### 训练 `asv_main.py`

1、选择目标轨迹类型

根据训练的目标轨迹类型，修改`asv_main.py`中的MAX_DECAYEP变量和target_trajectory

```python
MAX_DECAYEP = 1000
```

```python
env = ASVEnv(target_trajectory='linear')
```

- 直线轨迹：MAX_DECAYEP=1000；target_trajectory=‘linear'
- 曲线轨迹：MAX_DECAYEP=3000；target_trajectory=‘func_sin'

2、是否继续已有模型的训练

全新训练则`asv_main.py`中主函数`rl_loop()`不传入参数，使用默认参数

加载已有模型继续训练则`rl_loop('模型路径')`要传入模型的路径

```python
if __name__ == '__main__':
    rl_loop()	#全新训练
    rl_loop('./model/func_sin.pth')	   #加载已有模型
```

3、运行`asv_main.py`

- 控制台输出局次、本局累计奖励、本局步数信息

```
episode: 3426, cum_reward: -5.638307041911165, step_num:300
```

- tensorboard可视化a_loss、c_loss、cum_reward数据保存在`runs`文件夹下，命令行运行tensorboard进行查看
- 每局覆盖式保存训练模型在`model`文件夹下，直线轨迹为linear.pth，曲线轨迹为func_sin.pth
