#!/bin/bash

# 参数解析
DEVICE_NUM=2  # 默认使用2个设备
DEVICE_TARGET="Auto"  # 默认设备类型
DISTRIBUTE=true  # 是否使用分布式训练
EPISODE=1000  # 训练的总回合数

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -n|--device_num)
      DEVICE_NUM="$2"
      shift 2
      ;;
    -t|--device_target)
      DEVICE_TARGET="$2"
      shift 2
      ;;
    -d|--distribute)
      DISTRIBUTE="$2"
      shift 2
      ;;
    -e|--episode)
      EPISODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "启动并行训练，设备数量: $DEVICE_NUM, 设备类型: $DEVICE_TARGET, 分布式模式: $DISTRIBUTE, 训练回合: $EPISODE"

# 确保目录存在
mkdir -p log

# 如果使用分布式训练，启动多个容器
if [ "$DISTRIBUTE" = true ]; then
  for ((i=0; i<$DEVICE_NUM; i++))
  do
    echo "启动设备 $i ..."
    # 使用不同的容器名避免冲突
    CONTAINER_NAME="fighting_game_$i"
    # 使用不同的显示端口避免冲突
    DISPLAY_PORT=$((10+i))
    
    # 启动容器并运行训练
    sudo docker run -d --rm -v "${PWD}":/app -w /app --name $CONTAINER_NAME fighting_game:v1 /bin/sh -c "Xvfb :$DISPLAY_PORT -screen 0 1024x768x16 & export DISPLAY=:$DISPLAY_PORT; python train.py --device_target=$DEVICE_TARGET --parallel=true --distribute=true --device_num=$DEVICE_NUM --rank_id=$i --episode=$EPISODE > log/train_log_$i.txt 2> log/train_error_log_$i.txt" &
    
    # 等待一段时间，避免资源争用
    sleep 2
  done
  
  # 等待所有训练完成
  wait
  echo "所有训练进程已完成"
else
  # 本地并行模式，单个容器内进行
  sudo docker run -d --rm -v "${PWD}":/app -w /app --name fighting_game fighting_game:v1 /bin/sh -c "Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python train.py --device_target=$DEVICE_TARGET --parallel=true --distribute=false --device_num=$DEVICE_NUM --episode=$EPISODE > log/train_log.txt 2> log/train_error_log.txt"
fi