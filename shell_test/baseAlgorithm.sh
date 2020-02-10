#!/bin/bash

# 算术基本运算，加减乘除取余
a=11
b=5

# 加法 expr
val=`expr $a + $b`
echo "$a + $b = $val"

# 另一种数值运算 $[var]
val=$[a-b]
echo "$a - $b = $val"

# 除法
val=`expr $a / $b`
echo "$a / $b = $val"

# 关系运算
if [ $a -eq $b ]
then
    echo "$a -eq $b: a 等于 b"
else
    echo "$a -ne $b: a 不等于 b"
fi

# 布尔与逻辑运算
if [[ $a -gt 0 && $b -gt 0 ]]
then
    echo "a,b 都大于 0"
fi
