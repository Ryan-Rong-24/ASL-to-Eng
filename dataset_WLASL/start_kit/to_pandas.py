import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

content = json.load(open('WLASL_v0.3.json'))
VIDEO_PATH = 'raw_videos_mp4'

d = {'video_name':[],'tag':[]}
df = pd.DataFrame(data=d)

for entry in content:
    tag = entry['gloss']
    for inst in entry['instances']:
        video_id = inst['video_id']
        if os.path.exists(os.path.join(VIDEO_PATH,video_id+".mp4")):
            df.loc[len(df.index)] = [video_id+".mp4",tag]

train, test = train_test_split(df, test_size=0.1)
train.to_csv('train.csv')
test.to_csv('test.csv')
