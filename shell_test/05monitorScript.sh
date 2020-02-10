#!/bin/bash
# 服务器监控脚本
# 1.获取CPU的使用率，超过设定值，则报警；
# 2.获取内存的使用率，超过设定值，则报警；
# 3.获取磁盘容量可用率，小于设定值，则报警；
# 4.检查网站存活状态，异常则报警；
# 5.查看某进程（比如sshd）是否存在，异常则报警；
# 6.将以上结果写入日志；

# cpu 使用率 阈值
cpu_boundary=50
# 内存 使用率 阈值
memory_boundary=70
# 磁盘 使用率 阈值
disk_boundary=20
# 网络IP
network_ip="192.168.100.10"
# 网络状态
net_statu=0
# 进程 名称
ps_name=sshd
# 进程 存在的个数
progress_count=0

# 写入日志目录
log_path="./log/"
# 输出日志目录
log_output_file="output.log"
log_output_path="$log_path$log_output_file"
log_output_size=102400
# 告警日志目录
log_warning_file="warning.log"
log_warning_path="$log_path$log_warning_file"
log_warning_size=102400

now_time=$(date +'%Y-%m-%d %H:%M:%S')

# 检测 日志文件 是否存在
check_log(){
    local log_file_path=$1
    local log_output_file_path=$2
    if [ ! -d $log_file_path ]; then
	mkdir -p $log_file_path
    fi

    if [ ! -f $log_output_file_path ]; then
	touch "$log_output_file_path"
    fi

    if [ ! -r $log_output_file_path ]; then
	echo "$log_output_file_path cannot be read."
	return 1
    fi

    if [ ! -w $log_output_file_path ]; then
	echo "$log_output_file_path cannot be wrote."
	return 1
    fi
    
    return 0
}

# 日志文件大小 超过设定值，则备份
backup_log(){
    local log_file_path=$1
    local log_output_file_path=$2
    local log_file_limit_size=$3
    local log_file_size=`ls -l $log_output_file_path | awk '{print $5}'`
    # echo "backup log file: $log_file_path, $log_output_file_path, $log_file_limit_size, $log_file_size"
    if [ $log_file_size -gt $log_file_limit_size ]; then
	local back_file="$log_output_file_path`date +'%Y%m%d-%H%M%S'`.log"
	mv -f $log_output_file_path $back_file
	write_output_log "$log_output_file_path size is $log_file_size,greater than $log_file_limit_size.It was cpoied to $back_file."
	# echo "backup suc: $back_file" 
    fi
    return 0
}

# 写入 输出日志
write_output_log(){
    check_log $log_path $log_output_path
    if [ $? -ne 0 ]; then
	return 1
    fi

    backup_log $log_path $log_output_path $log_output_size
    echo "[$now_time] $1" | tee -a ${log_output_path}
    return 0
}

# 写入 告警日志
write_warning_log(){
    check_log $log_path $log_warning_path
    if [ $? -ne 0 ]; then
	return 1
    fi

    backup_log $log_path $log_warning_path $log_warning_size
    echo -e  "\033[31;5m[$now_time] $1\033[0m" | tee -a ${log_warning_path}
    return 0
}

# 获取当前CPU的使用率
check_cpu_usage_rate(){
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
    local cpu_used_rate=$(awk 'BEGIN{printf "%.2f\n",'$[cpu_used*100]'/'$cpu_total'}')
    write_output_log "CPU Total: $cpu_total, Used: $cpu_used, Usage Rate: $cpu_used_rate%."
    if [ $(echo "$cpu_used_rate > $cpu_boundary" | bc) -eq 1 ]; then
	write_warning_log "Warning: CPU usage rate is $cpu_used_rate%,greater than $cpu_boundary%."
	return 1
    fi
    return 0
}

# 获取内存使用率
check_memory_uasge_rate(){
    # 总内存大小
    local mem_total=`free -m | sed -n '2p' | awk '{print $2}'`
    # 已使用内存
    local mem_used=`free -m | sed -n '2p' | awk '{print $3}'`
    # 剩余内存
    local mem_free=`free -m | sed -n '2p' | awk '{print $4}'`

    # 使用内存百分比
    local mem_used_rate=$(awk 'BEGIN{printf "%.2f",'$[mem_used*100]'/'$mem_total'}')
    # 剩余内存百分比
    local mem_free_rate=$(awk 'BEGIN{printf "%.2f",'$[mem_free*100]'/'$mem_total'}')
    
    write_output_log "Memory Total: $mem_total, Used: $mem_used, Free: $mem_free, Usage rate: $mem_used_rate%, Free rate: $mem_free_rate%." 
    if [ $(echo "$mem_used_rate > $memory_boundary" | bc) -eq 1 ]; then
	write_warning_log "Warning: Memory usage rate is $mem_used_rate%,greater than $memory_boundary%."
	return 1
    fi
    return 0
}

# 获取磁盘容量可用率
check_disk_free_rate(){
    for info in $(df -P | grep /dev | awk '{print $1","$5}' | grep dev)
    do
	local disk_info=(${info//,/ })
	local disk_name=${disk_info[0]}
	local disk_used_rate=$(echo ${disk_info[1]} | sed 's/%//g')
	local disk_free_rate=$(expr 100 - $disk_used_rate)
	write_output_log "Disk Name: $disk_name, Used rate: $disk_used_rate%, Available rate: $disk_free_rate%."
	if [ $(echo "$disk_free_rate < $disk_boundary" | bc) -eq 1 ]; then
	    write_warning_log "Warning: Disk($disk_name) available rate is $disk_free_rate%,lower than $disk_boundary%."
	fi
    done
    return 0
}

# 检查网络是否可用
check_network(){
    local ip=$1
    if [ -z "$ip" ]
    then
	write_warning_log "You should input the checked IP."
	return
    fi

    ping -c1 $ip &>/dev/null
    [ $? -eq 0 ] && write_output_log "$ip on." || write_warning_log "$ip off."
    # [ $? -eq 0 ] && echo "$ip on." || echo "$ip off."
    return 0
}

# 检查某进程（比如sshd）是否存在
check_process(){
    local psid=$(ps -e | grep $ps_name | awk '{print $1}' | xargs)
    local psid_array=(${psid// / })
    local ps_count=${#psid_array[@]}
    write_output_log "$ps_name has $ps_count."
    if [ $ps_count -eq $progress_statu ]; then
	write_warning_log "$ps_name is not exist."
    fi
    return 0
}

count=0
while [ $count -lt 20 ]
do
    check_cpu_usage_rate &
    check_memory_uasge_rate &
    check_disk_free_rate &
    check_network $network_ip &
    check_process &
    count=$[count+1]
done

# 并发
wait
echo "Done!"
