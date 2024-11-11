# yolov8_colab_image

<b> 사진을 1.jpg로 만들고, yolov8에서 image detect를 colab에서 해보았다.
``` bash
# 필요한 패키지 설치
%pip install ultralytics opencv-python-headless matplotlib

# 파일 업로드를 위한 라이브러리 import
from google.colab import files

# 이미지 파일 업로드
uploaded = files.upload()
filename = next(iter(uploaded)) # 업로드된 파일 이름을 가져옵니다.

# 업로드된 이미지 파일을 적절한 디렉토리로 이동
!mkdir -p /content/sample_data   # 디렉토리가 없는 경우 생성
!mv {filename} /content/sample_data/{filename}

from ultralytics import YOLO
from IPython.display import Image, display
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
# Define path to the image file
image_path = '/content/sample_data/1.jpg'
# Run inference on the source
results = model.predict(image_path)  # list of Results objects
# Assuming results[0] contains the desired Result object
first_result = results[0]
first_result.save()  # Make sure to check the correct path of the saved image
saved_image_path = '/content/results_1.jpg'  # This path might change depending on your environment and run
display(Image(saved_image_path))
```
![results_1](https://github.com/user-attachments/assets/37a7c10f-efbb-450a-bac4-377bebbebcd7)
