## HAI x YBIGTA Hackathon baseline code
UIIS dataset에 대해 instance segmentation을 진행하기 위한 baseline code입니다.

baseline에서 사용한 모델은 Mask RCNN 모델입니다.

- MS COCO dataset에 대해 pretrained
- UIIS에 대해서는 학습 x

기본적인 segmentation만 진행했고 결과는 reults 폴더 안에 존재합니다

- - - - - - - - - 
### 코드 돌려보실 때
data 폴더를 만들고 그 안에 datasets을 넣어주세요!
```
--data
  |--annotations
    |--val.json
  |--val
--baseline.py
```