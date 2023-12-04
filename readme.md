**Installation and Usage**

`cd opt`

`pip install -r requirements`

`python preprocess.py --first_frame_path  path_to_your_first_image`

For example:

`python preprocess.py --first_frame_path data/imgs_1_trim/1_trim_frame_60.png`

`./run_track.sh  template_json_file_produced_by_the_last_step   images_folder_to_be_detected`

For example:
` ./run_track.sh runs/predicted_labels_1701644122619.json  data/imgs_286/  `


