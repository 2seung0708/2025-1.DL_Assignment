# 2025-1.무인항공시스템을 위한 딥러닝개론

해당 코드는 https://github.com/zhu-gl-ux/Adapter를 기반으로 작성되었습니다.

#### 실험 결과 정량적 리포팅
- 15-5: 15개의 base 이미지로 학습한 후, 5개의 class에 대해 추가 학습. 총 2번 학습
- 15-1: 15개의 base 이미지로 학습한 후, 1개의 class에 대해 추가 학습 반복. 총 6번 학습
- old average는 base class에 대한 mIoU 평균, new average 는 추가된 new class에 대한 mIoU 평균을 리포팅한 것


<img src="https://github.com/2seung0708/2025-1.DL_Assignment/blob/main/figures/results_table.png" width="700" height="600" />

## Requirements
- python 3.10
- torch==1.12.1
- torchvision==0.13.1
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib

## Setting

```bash
conda create -n adapter python=3.10 -y
conda activate adapter 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt
```

- PascalVOC 데이터를 아래의 경로에서 다운받아 준비합니다.
  - [🔗PascalVOC 데이터](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
  - 추가로 필요한 spilt에 대한 txt 파일을 다운받아  `{VOC2012}/split` 경로 안에 넣어둡니다.


## Training 

```bash
## 15-1 학습 코드
bash scripts/voc/overlapped/train_voc_15-1.sh

## 15-5 학습 코드
bash scripts/voc/overlapped/train_voc_15-5.sh
```

## Test

```bash
ckp={checkpoint 경로/checkpoint-epoch60.pth}
python eval_voc.py --device 0 --test --resume $cpt
```

- checkpoint를 아래의 경로에서 다운받아 준비합니다.
  - [🔗Checkpoint 다운로드](https://drive.google.com/file/d/1xt6xUuW0IAk5k0DgPtL1vtm0jW6qn0ou/view?usp=sharing)


## Citation
```
@inproceedings{zhu2025adaptive,
  title={Adaptive prototype replay for class incremental semantic segmentation},
  author={Zhu, Guilin and Wu, Dongyue and Gao, Changxin and Wang, Runmin and Yang, Weidong and Sang, Nong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10932--10940},
  year={2025}
}
```
