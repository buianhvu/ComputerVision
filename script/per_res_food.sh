#!/usr/bin/env bash

STATE_DIR=./output/res_101_food
STATE_LOG=./output/res_101_food
DATA=./Food-11/

if [ ! -d "$STATE_DIR" ]; then
    mkdir ${STATE_DIR}
fi


if [ ! -d "$STATE_LOG" ]; then
    mkdir ${STATE_LOG}
fi

cmd="python3.5 run_res.py -i ${DATA} -o ${STATE_DIR} --epoch 20 --opt-func adam --lr 0.0001 --model-name res_101_food --plog ${STATE_LOG} --batch-size 40"
eval ${cmd}

