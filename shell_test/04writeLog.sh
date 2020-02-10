#!/bin/bash

# 写日志
# 1. 日志文件：./tmp/my.log

# 2. 存在与权限检查
# * 判断 tmp 目录是否存在，不存在则新建
# * 判断 my.log 文件是否存在，不存在则新建
# * 判断 是否具备 读写权限

# 3. 备份检查
# * 如果日志文件大小超过1K，自动备份

# 4. 在文件尾追加信息，带上日志时间

log_path="./tmp/"
log_output_file="output.log"
log_output_path="$log_path$log_output_file"
log_output_size=100
log_warning_file="warning.log"
log_warning_path="$log_path$log_warning_file"
log_warning_size=100

check_log(){
    if [ ! -d $1 ]; then
	mkdir -p $1
    fi

    if [ ! -f $2 ]; then
	touch "$2"
    fi

    if [ ! -r $2 ]; then
	echo "$2 cannot be read."
	return 1
    fi

    if [ ! -w $2 ]; then
	echo "$2 cannot be wrote."
	return 1
    fi
    
    echo "$2 existed. It can be read and wrote."
    return 0
}

backup_log(){
    local size=`ls -l $2 | awk '{print $5}'`
    if [ $size -gt $3 ]; then
	local back_file="$2`date +'%Y%m%d-%H%M%S'`.log"
	mv -f $2 $back_file
	echo "$2 was cpoied to $back_file."
    fi
    return
}

# 写入告警日志
write_warning_log(){
    check_warning_path=$(check_log $log_path $log_warning_path)
    if [ $? -ne 0 ]; then
	return 1
    fi

    backup_warning_log=$(backup_log $log_path $log_warning_path $log_warning_size)
    echo "Warning: $1 is $2" | tee -a ${log_warning_path}
    return 0
}

write_warning_log "CPU usage rate" 0.45
write_warning_log "CPU usage rate" 0.5
write_warning_log "CPU usage rate" 45
write_warning_log "CPU usage rate" 80.5
