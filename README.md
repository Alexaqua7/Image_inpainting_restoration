# 이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발
이미지 색상화 및 손실 부분 복원 AI 경진대회를 위한 Repository입니다.

이미지 복원 기술은 손상되거나 결손된 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 기술로, 역사적 사진 복원, 영상 편집, 의료 이미지 복구 등 다양한 분야에서 중요하게 활용되고 있습니다.
이번 월간 데이콘에서는 이러한 복원 기술을 활용할 수 있는 Vision AI 알고리즘을 개발하는 것을 목표로 합니다. 
손실된 이미지의 특정 영역을 복구하고 흑백 이미지에 적합한 색을 입혀 원본과 유사한 이미지를 재창조하는 알고리즘을 만들어주세요!

## 설명
손실된 이미지의 결손 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 AI 알고리즘 개발

## 주최 / 주관
데이콘

## Project Structure
```
project_root/
│
├── data/
│   ├── test_input/
│   └── train_gt/
│     
│
├── Image_inpainting_restoration/
│   └── lightning
│      ├── train.py
│      ├── train_curriculum.py
│      ├── inference.py
│      └── utils/
│          ├── dataset.py
│          ├── dataset_1Q.py
│          ├── dataset_2Q.py
│          ├── dataset_3Q.py
│          ├── dataset_4Q.py
│          ├── model.py
│          ├── model_1Q.py
│          ├── model_2Q.py
│          ├── model_3Q.py
│          ├── model_4Q.py
│          ├── metric.py
│          └── split.py
│
│
├── requirements.txt
└── README.md
```
