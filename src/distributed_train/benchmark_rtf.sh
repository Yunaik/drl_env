#!/bin/bash


# for backend in 1
for backend in 0 1
do
    # for batch_size in 100
    for batch_size in 50
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
                    bash start_server.sh $amount_of_env & # start the servers in background
                    pid_start_server=$! # $! gets PID of last background process
                    echo "PID of start_server.sh:" $pid_start_server ". Sleeping for a 3s to let servers start"
                    echo -----------------------------------------------------------------------------
                    sleep 3
                    echo "Starting python parallel_batch.py"
                    echo -----------------------------------------------------------------------------

                    python3 parallel_batch.py --backend $backend --batch_size $batch_size --core $core --enlarge $enlarge --amount_of_env $amount_of_env 


                    echo "Finished running parallel_batch.py. Stopping server"
                    echo -----------------------------------------------------------------------------
                    bash stop_server.sh $amount_of_env
                    sleep 1
                    echo "Kill start_server.sh process"
                    echo -----------------------------------------------------------------------------
                    kill $pid_start_server
                    echo ==================================End========================================
                done
            done
        done
    done
done
