import os

first_path = "./AdaIN/test.py"

#保持背景的人像迁移
second_path = "./Mask_RCNN/predict.py"

#保持人像的背景迁移
# second_path = "./Mask_RCNN/predict2.py"
interpreter_path = "/Users/zhujiaxin/miniforge3/envs/studydeeplearning/bin/python"

os.system(f'{interpreter_path} {first_path}')
os.system(f'{interpreter_path} {second_path}')





