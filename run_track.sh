#!/bin/bash

template_path=$1
input_folder=$2

# 打印结果
echo "template_json: $template_json"
echo "input_folder: $2"

# 循环处理照片
for i in $(seq 0 2 4); do
    filename="${input_folder}286_frame_${i}.png"
    if [ -e "$filename" ]; then
        python track.py --yolo-model yolov8n --source "$filename" --template "$template_path"
    else
        echo "File not found: $filename"
    fi
done

