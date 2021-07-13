#!/bin/bash
# trap "kill 0" EXIT

if [ -z "$2" ]
  then
    gpu=0
else
    gpu=$2
fi
echo Running on GPU$gpu
export PHYSX_GPU_DEVICE=$gpu

let gpu_port=$gpu+5
# echo "Port:" "$gpu_port"00x
sleep 1
counter=0
amount=$1
while [ $counter -lt $amount ]
do
# echo $counter
echo Connecting server to port "$gpu_port"00$counter 
python3 env_server.py --addr 127.0.0.1:"$gpu_port"00$counter &
((counter++))
done
python3 env_server.py --addr 127.0.0.1:6000 & 
wait