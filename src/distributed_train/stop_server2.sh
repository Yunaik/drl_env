#!/bin/bash

lsof -ti :5000|xargs kill -15
lsof -ti :6000|xargs kill -15
