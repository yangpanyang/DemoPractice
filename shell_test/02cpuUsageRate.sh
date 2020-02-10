#!/bin/bash

# 获取当前CPU的使用率
cpu_usage_rate(){
    local line=$(cat /proc/stat | grep -B1 -m1 "cpu" | awk '{print}')
    local user=$(echo $line | awk '{print $2}')
    local nice=`echo $line | awk '{print $3}'`
    local system=`echo $line | awk '{print $4}'`
    local idle=`echo $line | awk '{print $5}'`
    local iowait=`echo $line | awk '{print $6}'`
    local irq=`echo $line | awk '{print $7}'`
    local softirq=`echo $line | awk '{print $8}'`
    local steal_time=`echo $line | awk '{print $9}'`
    local guest=`echo $line | awk '{print $10}'`
    local cpu_total=$[user+nice+system+idle+iowait+irq+softirq+steal_time+guest]
    local cpu_used=$[user+nice+system+irq+softirq+steal_time+guest]

    # $[cpu_used/spu_total] 默认为整数
    # 保留 2位精度
    echo "CPU Total: $cpu_total"
    echo "CPU Used: $cpu_used"
    local rate=$(awk 'BEGIN{printf "%.2f\n",'$[cpu_used*100]'/'$cpu_total'}')
    echo "CPU Usage: $rate%"
}

cpu_usage_rate
