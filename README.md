# Lumbar-Spine-Degeneration-classification
Lumbar Spine Degeneration commonly called as Low back pain is one of the leading causes of
disability worldwide. So, to solve these issues we use deep learning models to predict lumbar spine
degeneration faster in order to diagnose and grading of these conditions which help us guide
treatment and potential surgeries for the patients.
Our model uses magnetic resonance imaging (MRI) to classifying of five lumbar spine degenerative
conditions:
*Left Neural Foraminal Narrowing,
*Right Neural Foraminal Narrowing,
*Left Subarticular Stenosis,
*Right Subarticular Stenosis
*Spinal Canal Stenosis.
For each Magnetic resonance imaging (MRI) images, we’ve provided severity scores (Normal/Mild,
Moderate, or Severe) for each of the five conditions across the intervertebral disc levels L1/L2,
L2/L3, L3/L4, L4/L5, and L5/S1.

![Left Neural Foraminal Narrowing CNN](https://github.com/user-attachments/assets/472f7804-c60c-4ca3-ba18-135c6e79eb9d)

Model takes MRI images as input and gives us severity level as output. the model will consist of a
total of 7 layers in which 4 are convolution layer and another 3 is linear layer in which the sigmoid
function will be applied on the result of the last linear layer. The learning rate of the model will be
reduced according to the training to avoid over training of the model.
Before training the model, we appley HSV filter to the spine MRI image which will help the model
identifying the deformity in the spine faster and efficiently by enhancing the color spectrum of the
deeper parts of the spine MRI.
![130_right_subarticular_stenosis_l3_l4](https://github.com/user-attachments/assets/771b69da-11c7-4038-b589-9c48591d2557)
![267_spinal_canal_stenosis_l2_l3](https://github.com/user-attachments/assets/d97c4e48-deb3-442e-9e1d-c3b80c7f4b4a)
![772_right_neural_foraminal_narrowing_l4_l5](https://github.com/user-attachments/assets/07e00617-c5bb-4364-a12d-0ad4311f115a)
