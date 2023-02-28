# ASLtoEng
Translating real time video of American Sign Language (ASL) to English by Audible Motion.

## How it works
We use a pretrained solution (mediapipe) to detect landmarks of the person from the video. These landmarks are then stacked into a frame stack and passed to a LSTM model to learn the ASL motion. Finally, the model produces the English translation.
<br>
![Flow Chart](/images/flowchart.png)

## Web App
We use Flask to deploy our model on a web application:
<br>
![Web App](/images/webapp.png)

## Datasets
- WLASL (https://dxli94.github.io/WLASL/)
- MSASL (https://www.microsoft.com/en-us/research/project/ms-asl/)
<br>
note: the datasets are not on this repo since they are too large

By Audible Motion  <br>
![logo](/images/audiblemotionlogo.png)
