#!/bin/bash

# 1. 数组的定义
arr=(aa bb cc "hello world")

# 2. 设置 元素
arr[2]="222"

# 3. 读取 元素
echo "下标为0的元素："${arr[0]}
echo "下标为1的元素："${arr[1]}
echo "下标为2的元素："${arr[2]}
echo "下标为3的元素："${arr[3]}

# 4. 读取 所有元素
echo "all array info: "${arr[@]}

# 5. 获取数组长度
len=${#arr[@]}
echo "the length of array: "$len
