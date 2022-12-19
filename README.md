# 주제 : 애니메이션 얼굴 랜드마크 탐지

- Live 2d 젠이츠 이미지에 Facial Landmark Detecting한 결과 (사전학습된 모델 사용)
  
![Inception](https://user-images.githubusercontent.com/66348480/208411504-46af7e48-b604-4d1e-9b91-41efe9191fba.gif)

### GIF이미지에서 얼굴 랜드마크를 디텍팅한 이미지를 다운로드받을 수 있는 시스템을 제작했습니다. 

## 학습 방법
- Model : Resnet을 Backbone으로 하는 Inception Network를 사용합니다. 
- Data :
  1. 68개의 얼굴 랜드마크 포인트를 가지고 있는 Caricature Image dataset 약 6000장을 사전학습 모델에 사용합니다. 
  2. 60개의 얼굴 랜드마크 포인트를 가지고 있는 Manga109 데이터셋을 사용하여 학습합니다. 이 떄 1.의 학습가중치를 초기화하여 사용합니다. 

- Data Matching Program : 
    - Manga109와 landmark를 Matching하는 프로그램 (manga109_matching_program.ipynb)
      1. 전체 이미지에서 후보 이미지 선별 
      2. 후보 이미지 선별 과정 
         1. 이미지와 랜드마크 사이의 여백 제거 
         2. 조정한 랜드마크 위치가 음수인 경우 제거 (scale이 안맞는 이미지 제거)
         3. min이 전체 이미지 1/3지점보다 큰 이미지 제거 (scale이 안맞는 이미지 제거)
      3. 이미지 랜드마크 매칭
         1. 선별된 이미지를 Plot하여 수작업으로 고르고, x,y좌표를 수동조정해서 랜드마크 위치를 정확하게 매칭
    - 결과 : 400-450장의 이미지와 랜드마크 쌍 데이터를 확보 
* Manga109와 Landmark 파일은 저작권상 제공할 수 없습니다. 필요하다면 해당 사이트에 직접 방문하여 요청해야 합니다. 

## 사용법
### Download pretrained model
- 먼저 pretrained 모델을 받아줍니다.(model_new99.pt) 이 모델은 약 7800개의 캐리커쳐 이미지와 Manga109이미지들을 사전학습한 모델입니다. Inception Network를 이용합니다. 
### MakeGIF.py 실행
- GIF이미지와 사전학습된 모델의 가중치를 입력으로 받아 얼굴 랜드마크 탐지가 적용된 GIF를 다운로드 받아볼 수 있습니다. 
```sh- 
python MakeGIF.py --ModelWeightPath=model_new99.pt
                  --GIFPath=젠이치.gif
                  --OutputPath=outputPath
```

## 결과 비교 
-  Xception : Caricature Data 학습 Weight의 Entry block만 초기화하여 Manga Data 학습
-  Resnet50 : Caricature Data 학습 Weight의 전부를 초기화하여 Manga Data 학습
-  Inception : Caricature Data 학습 Weight의 전부를 초기화하여 Manga Data 학습

|Xception|Resnet50|Inception|
|--------|--------|---------|
![Xception](https://user-images.githubusercontent.com/66348480/208411301-d0ddcfc2-f933-4123-95b3-78dfb4a1ecd1.gif)|![Resnet50](https://user-images.githubusercontent.com/66348480/208414088-b435a3a2-21d1-449a-a042-13de2546863c.gif)|![Inception](https://user-images.githubusercontent.com/66348480/208415893-191fd05a-3672-4ca3-88d7-567d06695f34.gif)

## Inference Youtube
-  Xception : https://youtube.com/shorts/Z8F9Qo7AgeE?feature=share
-  Resnet50 : https://youtube.com/shorts/2m9TmERGR78
-  Inception : https://youtube.com/shorts/ko7mzJfTxfw

## 참고자료

- [1] Cai, Hongrui and Guo, Yudong and Peng, Zhuang and Zhang, Juyong (2021) Landmark detection and 3D face reconstruction for caricature using a nonlinear parametric model [https://github.com/Juyong/CaricatureFace](https://github.com/Juyong/CaricatureFace)
- [2] Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta (2020) Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications [http://www.manga109.org](http://www.manga109.org)
- [3] Marco Stricker, Olivier Augereau, Koichi Kise, Motoi Iwata (2018) Facial Landmark Detection for Manga Images [https://github.com/oaugereau/FacialLandmarkManga](https://github.com/oaugereau/FacialLandmarkManga)
- [4] Rishik Mourya (2020) Facial Landmarks Detection Using Xception Net [https://github.com/braindotai/Facial-Landmarks-Detection-Pytorch[Source Code] ](https://github.com/braindotai/Facial-Landmarks-Detection-Pytorch)


