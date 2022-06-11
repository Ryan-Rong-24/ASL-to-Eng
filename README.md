# ASLtoEng
Translating real time video of American Sign Language (ASL) to English <br>

Using mediapipe and a LSTM model for landmark recognition and sentence translation <br>

Trained using these datasets: <br>

## WLASL dataset
https://github.com/dxli94/WLASL <br>
- Follow steps to download <br>
- (might need to download linux on windows to run a bash script inside WLASL. Download WSL2 and Ubuntu distribution, might need to enable virtualization on windows, might need to do ```sed -i 's/\r$//' filename``` to solve windows line break errors) <br>
- when running preprocess.py, might need to install ffmpeg on WSL2: <br>
  ```sudo add-apt-repository ppa:mc3man/trusty-media
  sudo apt-get update
  sudo apt-get dist-upgrade
  sudo apt-get install ffmpeg```
