#!/bin/bash

bash start_server.sh 3 & # start the servers in background

$pid_start_server = $! # $! gets PID of last background process
echo $pid_start_server "PID"
echo "sleep 2 seconds"
sleep 4 # sleep 2

echo "Stopping server"
bash stop_server.sh 3
sleep 1
echo "Kill server"
kill $pid_start_server