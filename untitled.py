import cv2
import os,pandas
import shutil

def image_resizer():
	for i in [[f"{i}/{j}"for j in os.listdir(i)] for i in ["train_wolf/subarticular_stenosis/normal","train_wolf/subarticular_stenosis/Moderate",
	"train_wolf/subarticular_stenosis/Severe"]]:
		for i in i:
			cv2.imwrite(i,cv2.resize(cv2.imread(i),(512,512)))

image_resizer()

