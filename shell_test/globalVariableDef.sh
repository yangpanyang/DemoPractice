#!/bin/bash

# 1. export 导出一个环境变量
export MY_NAME="Caroline"
echo "global varibal ${MY_NAME}"

# 2. 查找自定义的环境变量
env | grep MY_NAME
