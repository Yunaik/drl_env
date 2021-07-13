#!/bin/bash

for gpu in 1
do
    for backend in 1
    # for backend in 0 1
    do
        for batch_size in 100 200
        # for batch_size in 100
        do
            for core in 8
            do
                # for enlarge in 1 3 5 10 20
                for enlarge in 3
                do
                    # for amount_of_env in 2
                    for amount_of_env in 10
                    do
                        echo ===============================Start===========================================
                        echo Running physx backend: $backend, batch_size: $batch_size, core: $core, enlarge: $enlarge, amount_of_env: $amount_of_env
                        echo -----------------------------------------------------------------------------

                        if [ $gpu -eq 4 ]
                        then
                            bash start_server.sh $amount_of_env 0 & # start the servers in background
                            pid_start_server0=$! # $! gets PID of last background process
                            sleep 3
                            bash start_server.sh $amount_of_env 1 & # start the servers in background
                            pid_start_server1=$! # $! gets PID of last background process
                            sleep 3

                            bash start_server.sh $amount_of_env 2 & # start the servers in background
                            pid_start_server2=$! # $! gets PID of last background process
                            sleep 3

                            bash start_server.sh $amount_of_env 3 & # start the servers in background
                            pid_start_server3=$! # $! gets PID of last background process
                        elif [ $gpu -eq 2 ]
                        then
                            bash start_server.sh $amount_of_env 0& # start the servers in background
                            pid_start_server0=$! # $! gets PID of last background process
                            sleep 3

                            bash start_server.sh $amount_of_env 1& # start the servers in background
                            pid_start_server1=$! # $! gets PID of last background process
                        elif [ $gpu -eq 1 ]
                        then
                            bash start_server.sh $amount_of_env & # start the servers in background
                            pid_start_server=$! # $! gets PID of last background process
                        else
                            echo "Not specified for" $gpu "gpus"
                        fi

                        echo "PID of start_server.sh:" $pid_start_server ". Sleeping for a 3s to let servers start"
                        echo -----------------------------------------------------------------------------
                        sleep 3
                        echo "Starting python parallel_batch.py"
                        echo -----------------------------------------------------------------------------

                        python3 parallel_batch.py --backend $backend --batch_size $batch_size --core $core --enlarge $enlarge --amount_of_env $amount_of_env --amount_of_gpu $gpu


                        echo "Finished running parallel_batch.py. Stopping server"
                        echo -----------------------------------------------------------------------------

                        if [ $gpu -eq 4 ]
                            then
                            bash stop_server.sh $amount_of_env 0
                            bash stop_server.sh $amount_of_env 1
                            bash stop_server.sh $amount_of_env 2
                            bash stop_server.sh $amount_of_env 3
                            sleep 2
                            echo "Kill start_server.sh process"
                            echo -----------------------------------------------------------------------------
                            kill $pid_start_server0 
                            kill $pid_start_server1
                            kill $pid_start_server2
                            kill $pid_start_server3
                        elif [ $gpu -eq 2 ]
                            then
                            bash stop_server.sh $amount_of_env 0
                            bash stop_server.sh $amount_of_env 1
                            sleep 2
                            echo "Kill start_server.sh process"
                            echo -----------------------------------------------------------------------------
                            kill $pid_start_server0 
                            kill $pid_start_server1
                        elif [ $gpu -eq 1 ]
                            then
                            bash stop_server.sh $amount_of_env
                            sleep 1
                            echo "Kill start_server.sh process"
                            echo -----------------------------------------------------------------------------
                            kill $pid_start_server
                        else
                            echo "Not specified for" $gpu "gpus"
                        fi

                        echo ==================================End========================================
                        sleep 1
                    done
                done
            done
        done
    done

done