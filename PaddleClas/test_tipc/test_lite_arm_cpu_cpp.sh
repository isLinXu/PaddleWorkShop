#!/bin/bash
source test_tipc/common_func.sh
current_path=$PWD

IFS=$'\n'

TIPC_CONFIG=$1
tipc_dataline=$(cat $TIPC_CONFIG)
tipc_lines=(${tipc_dataline})

work_path="./deploy/lite"
cd ${work_path}

BASIC_CONFIG="config.txt"
basic_dataline=$(cat $BASIC_CONFIG)
basic_lines=(${basic_dataline})

# parser basic config
label_path=$(func_parser_value_lite "${basic_lines[1]}" " ")
resize_short_size=$(func_parser_value_lite "${basic_lines[2]}" " ")
crop_size=$(func_parser_value_lite "${basic_lines[3]}" " ")
visualize=$(func_parser_value_lite "${basic_lines[4]}" " ")
enable_benchmark=$(func_parser_value_lite "${basic_lines[9]}" " ")
tipc_benchmark=$(func_parser_value_lite "${basic_lines[10]}" " ")

# parser tipc config
runtime_device=$(func_parser_value_lite "${tipc_lines[0]}" ":")
lite_arm_work_path=$(func_parser_value_lite "${tipc_lines[1]}" ":")
lite_arm_so_path=$(func_parser_value_lite "${tipc_lines[2]}" ":")
clas_model_name=$(func_parser_value_lite "${tipc_lines[3]}" ":")
inference_cmd=$(func_parser_value_lite "${tipc_lines[4]}" ":")
num_threads_list=$(func_parser_value_lite "${tipc_lines[5]}" ":")
batch_size_list=$(func_parser_value_lite "${tipc_lines[6]}" ":")
precision_list=$(func_parser_value_lite "${tipc_lines[7]}" ":")

LOG_PATH=${current_path}"/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"

#run Lite TIPC
function func_test_tipc(){
    IFS="|"
    _basic_config=$1
    _model_name=$2
    _log_path=$3
    for num_threads in ${num_threads_list[*]}; do
        if [ $(uname) = "Darwin" ]; then
            sed -i " " "s/num_threads.*/num_threads ${num_threads}/" ${_basic_config}
        elif [ $(expr substr $(uname -s) 1 5) = "Linux"]; then
            sed -i "s/num_threads.*/num_threads ${num_threads}/" ${_basic_config}
        fi
        for batch_size in ${batch_size_list[*]}; do
            if [ $(uname) = "Darwin" ]; then
                sed -i " " "s/batch_size.*/batch_size ${batch_size}/" ${_basic_config}
            elif [ $(expr substr $(uname -s) 1 5) = "Linux"]; then
                sed -i "s/batch_size.*/batch_size ${batch_size}/" ${_basic_config}
            fi
            for precision in ${precision_list[*]}; do
                if [ $(uname) = "Darwin" ]; then
                    sed -i " " "s/precision.*/precision ${precision}/" ${_basic_config}
                elif [ $(expr substr $(uname -s) 1 5) = "Linux"]; then
                    sed -i "s/precision.*/precision ${precision}/" ${_basic_config}
                fi
                _save_log_path="${_log_path}/lite_${_model_name}_runtime_device_${runtime_device}_precision_${precision}_batchsize_${batch_size}_threads_${num_threads}.log"
                real_inference_cmd=$(echo ${inference_cmd} | awk -F " " '{print path $1" "path $2" "path $3}' path="$lite_arm_work_path")
                command1="adb push ${_basic_config} ${lite_arm_work_path}"
                eval ${command1}
                command2="adb shell 'export LD_LIBRARY_PATH=${lite_arm_work_path}; ${real_inference_cmd}'  > ${_save_log_path} 2>&1"
                eval ${command2}
                status_check $? "${command2}" "${status_log}" "${model_name}"
            done
        done
    done
}


echo "################### run test tipc ###################"
label_map=$(echo ${label_path} | awk -F "/" '{print $NF}')

if [ $(uname) = "Darwin" ]; then
    # for Mac
    sed -i " " "s/runtime_device.*/runtime_device arm_cpu/" ${BASIC_CONFIG}
    escape_lite_arm_work_path=$(echo ${lite_arm_work_path//\//\\\/})
    sed -i " " "s/clas_model_file.*/clas_model_file ${escape_lite_arm_work_path}${clas_model_name}.nb/" ${BASIC_CONFIG}
    sed -i " " "s/label_path.*/label_path ${escape_lite_arm_work_path}${label_map}/" ${BASIC_CONFIG}
    sed -i " " "s/tipc_benchmark.*/tipc_benchmark 1/" ${BASIC_CONFIG}
elif [ $(expr substr $(uname -s) 1 5) = "Linux"]; then
    # for Linux
    sed -i "s/runtime_device.*/runtime_device arm_cpu/" ${BASIC_CONFIG}
    escape_lite_arm_work_path=$(echo ${lite_arm_work_path//\//\\\/})
    sed -i "s/clas_model_file.*/clas_model_file ${escape_lite_arm_work_path}${clas_model_name}/" ${BASIC_CONFIG}
    sed -i "s/label_path.*/label_path ${escape_lite_arm_work_path}${label_path}/" ${BASIC_CONFIG}
    sed -i "s/tipc_benchmark.*/tipc_benchmark 1/" ${BASIC_CONFIG}
fi
func_test_tipc ${BASIC_CONFIG} ${clas_model_name} ${LOG_PATH}
