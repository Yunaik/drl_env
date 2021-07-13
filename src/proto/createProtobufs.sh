#!/bin/bash

if [[ -f dist_pb2.py ]]
then
    rm dist_pb2.py
fi
if [[ -f ./dist_pb2_grpc.py ]]
then
    rm dist_pb2_grpc.py
fi

protoc --proto_path=. --python_out=. dist.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. dist.proto

cp dist_pb2.py ../distWrapper
cp dist_pb2_grpc.py ../distWrapper
