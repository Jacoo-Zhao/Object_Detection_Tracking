import time
import pandas as pd
import os
import json

# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import pdb
from functools import partial
from pathlib import Path

import torch
from ultralytics import YOLO

@torch.no_grad()
def run(args):

    # classes filter
    if args.classes is None and args.template:

        json_file_path = args.template
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # 提取 class_id 变量
        args.classes = data.get('class_id', None)

        # if args.classes is not None:
        #     print(f"\nClass_id: {args.classes}")
        # else:
        #     print("未找到 class_id 变量或其值为 None")


    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )
    results = yolo.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        # project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    # if 'yolov8' not in str(args.yolo_model):
    #     # replace yolov8 model
    #     m = get_yolo_inferer(args.yolo_model)
    #     model = m(
    #         model=args.yolo_model,
    #         device=yolo.predictor.device,
    #         args=yolo.predictor.args
    #     )
    #     yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    list = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            path = result.path
            class_name = yolo.names[cls]
            conf = int(box.conf[0] * 100)
            bx = box.xywhn.tolist()
            df = pd.DataFrame(
                {'path': path, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
            list.append(df)
    df = pd.concat(list, ignore_index=True) if list else pd.DataFrame()

    base_filename = 'predicted_labels'
    timestamp = int(time.time() * 1000)
    extension = 'csv'
    filename = f'{base_filename}_{timestamp}.{extension}'
    directory = 'runs/'
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f'Results saved to: {filepath}')
    return filepath


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--project', default=ROOT / 'runs' / 'track',
    #                     help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save_txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')

    parser.add_argument('--template', default='', type=str,
                        help='object template path')
    parser.add_argument("--first_frame_path", help="path to input first frame")
    parser.add_argument("--input_folder", help="Path to the input images folder")

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

