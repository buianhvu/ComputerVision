#!/usr/bin/env bash

STATE_DIR=/content/drive/My\ Drive/output/
STATE_LOG=/content/drive/My\ Drive/output/
DATA=/content/drive/Food-11/

if [ ! -d "$STATE_DIR" ]; then
    mkdir ${STATE_DIR}
fi


if [ ! -d "$STATE_LOG" ]; then
    mkdir ${STATE_LOG}
fi

cmd="python run_res -i DATA -o ${STATE_DIR} --epoch 10 --opt-func adam --lr 0.0001 --model-name res_101_food --plog ${STATE_LOG} --batch-size 40"
eval ${cmd}


