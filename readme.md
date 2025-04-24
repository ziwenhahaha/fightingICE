> mindspore_ac是助教的ac代码
mindspore_dqn是助教的dqn代码
torch_dqn是我重构的代码，里面实现好了并行

## 简单易用的使用说明：
把助教的DQN代码下载到本地，然后改名字为torch_dqn（改成什么都可以）

然后把这个项目的torch_dqn的所有文件，覆盖过去下载的项目那里

0、配环境
``` bash
# 1、安装anaconda
wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
# 2、以默认模式静默安装
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b

# 3、创建3.8的环境，docker的python版本就是3.8
conda create -n fighting python=3.8

# 4、激活并在虚拟环境中下载jdk11
conda activate fighting
conda install -c conda-forge openjdk=11  


# 5、这两个指令是设置虚拟显示器，没有这个，java的GUI会出不来从而报错
Xvfb :10 -screen 0 1024x768x16 &
export DISPLAY=:10

#【可选】6、cd到train.sh所在的目录，使用这个指令来测试java环境是否配好了
java  -cp FightingICE.jar:./lib/lwjgl/*:./lib/natives/linux/*:./lib/* Main --a1 BCP --a2 BlackMamba --c1 ZEN --c2 ZEN -n 1 --mute --fastmode --disable-window

# 7、下载pytorch
conda config --add channels conda-forge

conda install cudatoolkit=11.8 -c pytorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 8、安装myenv环境
pip install gym===0.15.3 py4j==0.10.9.5 port-for opencv-python==4.2.0.34

```



1、先执行
``` bash
Xvfb :10 -screen 0 1024x768x16 &
```
2、直接执行
``` bash
python train.py #在任何目录执行均可，已经把工作区路径重定向为train.py所在的目录
```