#!/bin/bash

runs_path="runs"

if [ ! -d "$runs_path" ]; then
    echo "The 'runs' path does not exist. Creating it now."
    mkdir "$runs_path"
else
    echo "The 'runs' path already exists."
fi

template_path=$1
input_folder=$2

# 打印结果
echo "template_json: $template_path"
echo "input_folder: $2"

# 循环处理照片
for i in $(seq 0 2 4); do
    filename="${input_folder}frame_${i}.png"
    if [ -e "$filename" ]; then
        python track.py --yolo-model yolov8n --source "$filename" --template "$template_path"
    else
        echo "File not found: $filename"
    fi
done

