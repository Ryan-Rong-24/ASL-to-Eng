# ASLtoEng
Translating real time video of American Sign Language (ASL) to English



WLASL dataset
https://github.com/dxli94/WLASL
Follow steps to donwload
(might need to download linux on windows to run bash script. Download WSL2 and Ubuntu distribution, might need to enable virtualization on windows, might need to do sed -i 's/\r$//' filename to solve windows line break errors)
when running preprocess.py, might need to install ffmpeg on WSL2: 
  sudo add-apt-repository ppa:mc3man/trusty-media
  sudo apt-get update
  sudo apt-get dist-upgrade
  sudo apt-get install ffmpeg
