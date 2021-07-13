#!/bin/bash


for backend in 1
# for backend in 0 1
do
    for batch_size in 1 10 50 100 200 250
    # for batch_size in 100
    do
        for core in 6 8 10 12 
        do
            # for enlarge in 1 3 5 10 20
            for enlarge in 3
            do
                # for amount_of_env in 2
                for amount_of_env in 1 2 4 8 10
                do
                    echo Running backend: $backend, batch_size: $batch_size, core: $core, enlarge: $enlarge, amount_of_env: $amount_of_env
                    bash start_server.sh $amount_of_env &
                    sleep 2
                    echo "Starting python"

                    python3 parallel_batch.py --backend $backend --batch_size $batch_size --core $core --enlarge $enlarge --amount_of_env $amount_of_env 
                    echo ==========================================================================
                done
            done
        done
    done
done

