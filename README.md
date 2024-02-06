# AI Tech 6기 Team 아웃라이어

## Members
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kangshwan">
        <img src="https://imgur.com/98RBpCF.jpg" width="100" height="100" /><br>
        강승환
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/viitamin">
        <img src="https://imgur.com/3fxWv0N.jpg" width="100" height="100" /><br>
        김승민
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/tjfgns6043">
        <img src="https://imgur.com/E19B6yJ.jpg" width="100" height="100" /><br>
        설훈
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/leedohyeong">
        <img src="https://imgur.com/Q4dLfWE.jpg" width="100" height="100" /><br>
        이도형
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/wjsqudrhks">
        <img src="https://imgur.com/W8gdsbD.jpg" width="100" height="100" /><br>
        전병관
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/seonghyeokcho">
        <img src="https://imgur.com/XYXAxcJ.jpg" width="100" height="100" /><br>
        조성혁
      </a>
    </td>
  </tr>
</table>

# 글자 검출 프로젝트
학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회
![OCR Image](https://imgur.com/gmfwpvO.jpg)

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 
또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 
이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.
이번 대회는 OCR의 대표적인 model 중 하나인 EAST model을 활용하여 진료비 계산서 영수증안에 있는 글자를 인식하는 대회입니다.

이번 대회는 Data-Centric 대회로 다음과 같은 제약사항이 있습니다.

- 대회에서 주어지는 EAST model만을 사용해야 하며 model과 관련된 코드를 바꿔서는 안됩니다.
- 이미지넷 기학습 가중치 외에는 사용이 불가합니다.

즉 이번 대회는 모델을 고정한 상태로 데이터만을 활용하여 OCR model의 성능을 최대한 끌어 올리는 프로젝트 입니다.

이번 대회는 부스트캠프 AI Tech CV 트랙내에서 진행된 대회이며 F1-Score로 최종평가를 진행하였습니다.

## Final Score
Public

![Public](https://imgur.com/C2G5qIx.jpg)

Private

![private](https://imgur.com/uMTFxJj.jpg)

## Ground Rules
### [Conventional Commits 1.0.0](https://www.conventionalcommits.org/ko/v1.0.0/)
```bash
<타입>[적용 범위(선택 사항)]: <설명>

[본문(선택 사항)]

[꼬리말(선택 사항)]
```

#### Types
- fix | feat | BREAKING CHANGE | build | chore | ci | docs | style | refactor | test | release
  - fix : 기능에 대한 버그 수정
  - feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
  - build : 빌드 관련 수정
  - chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
  - ci : CI 관련 설정 수정
  - docs : 문서(주석) 수정
  - style : 코드 스타일, 포맷팅에 대한 수정
  - refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
  - test : 테스트 코드 추가/수정
  - release : 버전 릴리즈

## Folder Structure
  ```
code
├── utils
|   └── ensemble
|       ├── coco2ufo.ipynb
|       ├── ensemble.ipynb
|       ├── lift_up_bounding_boxes.ipynb
|       ├── manual_k_fold.ipynb
|       ├── ufo2coco.ipynb
|       ├── visualization.ipynb
|       └── weighted_boxes_fusion.py
├── dataset.py
├── detect.py
├── deteval.py
├── east_dataset.py
├── inference.py
├── loss.py
├── metrics.py
├── model.py
├── train.py
└── requirements.txt
  ```
- ```utils``` 폴더에는 실험과 성능에 관련된 기능들을 넣어두었습니다.
  
|File(.ipynb/.py)|Description|
|---|---|
|coco2ufo|coco foramt으로 작성된 json file을 ufo format으로 변환합니다.|
|ensemble|wbf 알고리즘을 활용한 ensemble 코드입니다.|
|lift_up_bounding_boxes|supervisely에서 작업한 bbox가 int type으로 float정보가 손실되어 bbox가 맞지않아 보정해주는 코드입니다.|
|manual_k_fold|k-fold에 대한 설명과 실행이 적혀있는 코드입니다.|
|ufo2coco|ufo foramt으로 작성된 json file을 coco format으로 변환합니다.|
|visualization|json파일을 이용하여 Image와 Bounding box를 시각화하는 코드입니다.|
|weighted_boxes_fusion|WBF 코드입니다.|
## Dataset
- Total Images : 200장 (train : 100, test : 100)

# Wrap-Up Report
- [Wrap-Up-Report](https://synonymous-ton-89f.notion.site/Wrap-up-Reports-7d42ab3afbcc4e46ad10ec5365e3b2b5?pvs=4)

# Dataset 출처
- 대회에서 사용된 ```부스트캠프 AI Tech```임을 알려드립니다.

