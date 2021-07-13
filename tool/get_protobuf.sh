#!/bin/bash

rm -rf protobuf

#wget https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-osx-x86_64.zip

mkdir protobuf
#unzip protoc-3.7.1-linux-x86_64.zip -d protobuf/linux
unzip protoc-3.7.1-osx-x86_64.zip -d protobuf/darwin

rm protoc-3.7.1-*
