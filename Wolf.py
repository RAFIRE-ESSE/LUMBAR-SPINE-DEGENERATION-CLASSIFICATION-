import pandas 
import numpy,os
import pydicom,cv2

label=pandas.read_csv("label.csv")
train=pandas.read_csv("train.csv")

print(numpy.array(label["study_id"]),numpy.array(label["series_id"]))
if os.path.exists(f"train_wolf")==False:
		os.mkdir(f"train_wolf")
		os.mkdir(f"train_wolf/spinal_canal_stenosis")
		os.mkdir(f"train_wolf/spinal_canal_stenosis/normal")
		os.mkdir(f"train_wolf/spinal_canal_stenosis/Moderate")
		os.mkdir(f"train_wolf/spinal_canal_stenosis/Severe")
		os.mkdir(f"train_wolf/left_neural_foraminal_narrowing")
		os.mkdir(f"train_wolf/left_neural_foraminal_narrowing/normal")
		os.mkdir(f"train_wolf/left_neural_foraminal_narrowing/Moderate")
		os.mkdir(f"train_wolf/left_neural_foraminal_narrowing/Severe")
		os.mkdir(f"train_wolf/right_neural_foraminal_narrowing")
		os.mkdir(f"train_wolf/right_neural_foraminal_narrowing/normal")
		os.mkdir(f"train_wolf/right_neural_foraminal_narrowing/Moderate")
		os.mkdir(f"train_wolf/right_neural_foraminal_narrowing/Severe")
		os.mkdir(f"train_wolf/left_subarticular_stenosis")
		os.mkdir(f"train_wolf/left_subarticular_stenosis/normal")
		os.mkdir(f"train_wolf/left_subarticular_stenosis/Moderate")
		os.mkdir(f"train_wolf/left_subarticular_stenosis/Severe")
		os.mkdir(f"train_wolf/right_subarticular_stenosis")
		os.mkdir(f"train_wolf/right_subarticular_stenosis/normal")
		os.mkdir(f"train_wolf/right_subarticular_stenosis/Moderate")
		os.mkdir(f"train_wolf/right_subarticular_stenosis/Severe")

for i in zip(numpy.array(label["study_id"]),numpy.array(label["series_id"])
	,numpy.array(label['instance_number']),numpy.array(label['condition']),numpy.array(label['level']),range(len(numpy.array(label['level'])))):
	
	w_=f"{'_'.join(i[3].split(' '))}_{'_'.join(i[4].split('/'))}".lower()
	wolf=train[train["study_id"]==i[0]].to_dict('list')[w_][0]
	if wolf=="Normal/Mild":
		wolf="normal"

	print(f"{i[3]}/{wolf}/{w_}.jpg")
	if i[3]=="Spinal Canal Stenosis":
		cv2.imwrite(f"train_wolf/spinal_canal_stenosis/{wolf}/{i[5]}_{w_}.jpg",cv2.normalize(pydicom.dcmread(f"train_images/{i[0]}/{i[1]}/{i[2]}.dcm").pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
	if i[3]=="Right Neural Foraminal Narrowing":
		cv2.imwrite(f"train_wolf/right_neural_foraminal_narrowing/{wolf}/{i[5]}_{w_}.jpg",cv2.normalize(pydicom.dcmread(f"train_images/{i[0]}/{i[1]}/{i[2]}.dcm").pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
	if i[3]=="Left Neural Foraminal Narrowing":
		cv2.imwrite(f"train_wolf/left_neural_foraminal_narrowing/{wolf}/{i[5]}_{w_}.jpg",cv2.normalize(pydicom.dcmread(f"train_images/{i[0]}/{i[1]}/{i[2]}.dcm").pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
	if i[3]=="Left Subarticular Stenosis":
		cv2.imwrite(f"train_wolf/left_subarticular_stenosis/{wolf}/{i[5]}_{w_}.jpg",cv2.normalize(pydicom.dcmread(f"train_images/{i[0]}/{i[1]}/{i[2]}.dcm").pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
	if i[3]=="Right Subarticular Stenosis":
		cv2.imwrite(f"train_wolf/right_subarticular_stenosis/{wolf}/{i[5]}_{w_}.jpg",cv2.normalize(pydicom.dcmread(f"train_images/{i[0]}/{i[1]}/{i[2]}.dcm").pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))






