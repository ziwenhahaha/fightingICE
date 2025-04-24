> mindspore_ac是助教的ac代码
mindspore_dqn是助教的dqn代码
torch_dqn是我重构的代码，里面实现好了并行

## 简单易用的使用说明：
把助教的DQN代码下载到本地，然后改名字为torch_dqn（改成什么都可以）

然后把这个项目的torch_dqn的所有文件，覆盖过去下载的项目那里

然后：
1、先执行
``` bash
Xvfb :10 -screen 0 1024x768x16 &
```
2、直接执行
``` bash
python train.py #在任何目录执行均可，已经把工作区路径重定向为train.py所在的目录
```