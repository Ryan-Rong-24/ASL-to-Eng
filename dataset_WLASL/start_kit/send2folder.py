# creating folders 

import os
import json
import cv2

import shutil

root = 'C:\\Users\\ryanr\\ASLtoEng\\'
MP_DATA_PATH = os.path.join(root,'MP_data')


def send_to_folder(content):
    cnt = 1

    for entry in content:
        label = entry['gloss']
        label_path = os.path.join(MP_DATA_PATH,label)
        if not os.path.exists(label_path):
            os.makedirs(label_path, exist_ok=True)

        instances = entry['instances']

        for inst in instances:
            url = inst['url']
            video_id = inst['video_id']

            cnt += 1

            src_video_path = os.path.join(root, 'WLASL', 'start_kit', 'raw_videos_mp4_speedup', "fast_"+video_id + '.mp4')
            dst_video_path = os.path.join(label_path, "fast_"+video_id + '.mp4')

            if os.path.exists(dst_video_path):
                print('{} exists.'.format(dst_video_path))
                continue

            if not os.path.exists(src_video_path):
                continue

            print(cnt, dst_video_path)
            shutil.copyfile(src_video_path, dst_video_path)

        
def main():

    content = json.load(open('WLASL_v0.3.json'))
    send_to_folder(content)


if __name__ == "__main__":
    main()

