#!/bin/bash
counter=0
amount=$1


if [ -z "$2" ]
  then
    gpu=0
else
    gpu=$2
fi


let gpu_port=$gpu+5
echo "Killing port:" "$gpu_port"00x

while [ $counter -lt $amount ]
do
lsof -ti :"$gpu_port"00$counter|xargs kill -15
echo killed processes ending with "$gpu_port"00$counter
((counter++))
done
