#!/bin/bash
# 猜数字小游戏
# 1. 程序会随机从0-99产生一个数字，作为谜底
# 2. 用户输入可能数字
# 3. 程序判断，如果错误告知，是大还是小，用户可以根据提示继续猜；如果正确，告知猜测的次数，并退出程序

min=0
max=99
function random()
{
    min=$1;
    max=$2-$1;
    num=$(date +%s+%N);
    retnum=$[num%max+min];
    echo $retnum
}
ans=$(random $min $max)
echo -e "answer is $ans\n"

read -p  "Enter a number in ($min, $max) > " numberIn
inputCount=1

while true
do
    if [ $numberIn -eq $ans ]
    then
	echo -e  "\nAnswer is right! Input count is $inputCount"
	break
    elif [ $numberIn -gt $ans ]
    then
	max=$numberIn
	echo "Greater..."
    else
	echo "Lower..."
	min=$numberIn
    fi

    read -p "Enter a number in ($min, $max) > " numberIn
    inputCount=$[inputCount+1]
done;    
