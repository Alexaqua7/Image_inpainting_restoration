# 🖼️이미지 색상화 및 손실 부분 복원 AI 경진대회 

> 월간 데이콘
---

## 📌 대회 개요


이미지 복원 기술은 손상되거나 결손된 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 기술로, 역사적 사진 복원, 영상 편집, 의료 이미지 복구 등 다양한 분야에서 중요하게 활용되고 있습니다.
이번 월간 데이콘에서는 이러한 복원 기술을 활용할 수 있는 Vision AI 알고리즘을 개발하는 것을 목표로 합니다. 
손실된 이미지의 특정 영역을 복구하고 흑백 이미지에 적합한 색을 입혀 원본과 유사한 이미지를 재창조하는 알고리즘을 만들어주세요!

---

## 설명
손실된 이미지의 결손 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 AI 알고리즘 개발

---
## 주최 / 주관
데이콘

---
## 👥 팀원 소개

팀명: 건설용 자갈 암석

<table>
  <tr>
    <td align="center" style="padding: 12px;">
      <strong>DonghwanSeo</strong><br>
      <div style="height:1px; background-color:#ddd; width:60%; margin:6px auto;"></div>
      <img src="https://github.com/user-attachments/assets/f1a3b705-6e42-433e-9e00-9f9243d00c07" width="80"/>
    </td>
    <td align="center" style="padding: 12px;">
      <strong>aqua3g</strong><br>
      <div style="height:1px; background-color:#ddd; width:60%; margin:6px auto;"></div>
      <img src="https://github.com/user-attachments/assets/3d0a8319-2e5d-4add-93d9-131d6f2f9d97" width="80"/>
    </td>
  </tr>
</table>

---
## 🗂️ 데이터 소개

- **이미지 수**: 총 `29703`장  
  - train: 29603장  
  - test: 100장  


---
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
│      ├── original.py
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
