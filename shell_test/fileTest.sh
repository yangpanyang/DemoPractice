#!/bin/bash

# 文件的测试运算
file="baseAlgorithm.sh"

echo "文件是：$file"
if [ -e $file ]
then
    echo "文件存在"
fi

if [ -s $file ]
then
    echo "文件为空"
fi

if [ -f $file ]
then
    echo "是普通文件"
fi

if [ -x $file ]
then
    echo "文件可执行"
else
    echo "文件不可执行"
fi

bash file
