#!/usr/bin/env bash

STATE_DIR=/content/drive/My_Drive/output/
STATE_LOG=/content/drive/My_Drive/output/
DATA=/content/drive/Food-11/

if [ ! -d "$STATE_DIR" ]; then
    mkdir ${STATE_DIR}
fi


if [ ! -d "$STATE_LOG" ]; then
    mkdir ${STATE_LOG}
fi

cmd="python run_res -i DATA --model-name \"res_101_food\" -o ${STATE_DIR}"
eval ${cmd}


