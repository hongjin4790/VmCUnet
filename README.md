# VmCUnet

피부병변 영상 분할의 성능향상을 위한 VmCUnet [논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12057471)의 코드 저장소입니다. 


## Abstract

본 논문에서는 피부병변 영상에서 이미지 분할 성능을 향상시키기 위해 설계된 딥러닝 모델인 VmCUnet을 제안한다. VmCUnet은 Vm-UnetV2와 CIM(Cross-Scale Interaction Module)을 결합하여 인코더의 각 계층에서 추출한 특징들을 CIM으로 통합하여다양한 패턴과 경계를 정확하게 인식할 수 있다. VmCUnet은 ISIC-2017와 ISIC-2018 데이터 세트를 사용하여 피부 병변의 이미지 분할을 수행하였고 Unet, TransUnet, SwinUnet Vm-Unet, Vm-UnetV2와 비교하여 성능 지표인 IoU, Dice Score에서 더높은 성능을 보였다. 향후 작업에서는 다양한 의료 영상 데이터 세트에 대한 추가 실험을 수행하여 VmCUnet 모델의 일반화 성능을 검증할 예정이다

## Main Environments

```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Prepare the dataset

### datasets

- The ISIC2017 and ISIC2018 datasets 다운로드 후에 dataset_processing에 있는 코드 실행하면 train, valid, test 생성
- ./data/isic2017/
    - train
        - images
        - masks
    - valid
        - images
        - masks
    - test
        - images
        - masks
- ./data/isic2018/
    - train
        - images
        - masks
    - valid
        - images
        - masks
    - test
        - images
        - masks

## **Prepare the pre_trained weights**


- 사전 학습된 Vmamba의 가중치는 [here](https://github.com/MzeroMiko/VMamba) 다운로드 가능
- 다운로드 후에는 사전 학습된 가중치를 nets/pre_trained_weights/에 저장

## **Train the Model**

먼저, Config.py에서 모델, 데이터셋 경로, 하이퍼파라미터를 수정 후 training 코드 실행

```
python experiments/train_model.py
```

## **Evaluate the Model**

- Config.py에서 테스트 모델과 데이터셋 선택되었는지 확인

```
python experiments/test_model.py
```
