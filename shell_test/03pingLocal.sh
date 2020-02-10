#!/bin/bash
# 探测本地网络：探测某个ip地址是否通畅

test_ping()
{
    local ip=$1
    if [ -z "$ip" ]
    then
	echo "参数不得为空"
	return
    fi

    ping -c1 $ip &>/dev/null

    # if [ $? -eq 0 ]
    # then
    #     echo "$ip on"
    # else
    #     echo "$ip off"
    # fi
    [ $? -eq 0 ] && echo "$ip on" || echo "$ip off"
}

for ((i=1; i<255; i++))
do
    test_ping "192.168.100.$i" &
done

# 并发
wait
echo "Done!"
