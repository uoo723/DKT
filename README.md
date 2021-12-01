# Upstage Deep Coding Test

Task: Deep Knowledge Tracing

<img src='https://user-images.githubusercontent.com/7765506/144058077-a15db45e-beac-43b8-94ec-1cf98d10c68a.jpg' width=600 />

## Summary

Knowledge Tracing (KT)의 목적은 학습자의 과거 interaction을 기반으로 지식상태를 모델링하는 것이다. 본 프로젝트에서는 KT의 여러 task 중 학습자의 이전 문제 풀이 정보를 기반으로 아직 풀지 않은 문제의 정오답을 예측하는 task를 수행한다. 본 task를 수행했던 절차는 다음과 같다.
    
  * baseline 코드 돌리기. (LSTM 기반 모델)
  * SOTA 모델 서치 및 구현.
  * Data 전처리 변경.
  * 기타 성능 향상을 위한 training 최적화.
  * Hyperparameter optimization (HPO) 수행.
    
## Approach

### Dataset

![TrainDataset](https://user-images.githubusercontent.com/7765506/144045870-d03fc155-c39e-442a-9dda-448d7af6e394.png)

  * 5개의 feature와 1개의 정답 레이블 구성된 2,266,586개의 데이터.
  * DKT 모델에서 주로 공통적으로 사용하는 feature는 `assessmentItemID` (exercise), `KnowledgeTag` (knowledge concept) 이다.
  * Seq2seq 형태의 input으로 데이터를 DKT모델에 넣어줘야 하기 때문에 `userID`로 그룹핑함.

![user_id_n_seq](https://user-images.githubusercontent.com/7765506/144052036-59b488af-466b-47b0-977d-ec00ad148dad.jpg)

min. seq: 9; max. seq: 1,860; avg. seq: 338.40; std. seq: 321.31.

유저별로 문제를 푸는 양이 상이함.

  * 총 6,698개의 sequence를 7:3의 비율로 train:valid로 구성함.
  * 모델 학습을 위해 배치로 구성하기 위해 고정된 길이의 sequence로 구성해야 함.
    - 기존 베이스라인 코드에서는 `max_seq_len`보다 길면 잘라서 버리고, 짧으면 왼쪽에 zero padding으로 길이를 맞춤.
    - 데이터 소실을 줄이기 위해 `max_seq_len`보다 길면 partition하여 새로운 sequence로 구성.

### Model Architectures

#### LSTM

<img src='https://user-images.githubusercontent.com/7765506/144058917-dca28b53-ca90-437e-9578-246c835d0893.jpg' width=600 />

  * Sequence input에 간단하게 적용해볼 수 있는 RNN-based 모델.
  * 4개의 embedding vector를 concat하여 fc layer를 통과시켜 만든 hidden representation을 LSTM에 feed함.

#### SAKT ([A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837.pdf), EDM'19)

<img src='https://user-images.githubusercontent.com/7765506/144059682-094b177c-95dd-4c46-bc9c-6fd47afdacfe.jpg' width=600 />

  * Multi-head attention 모델.
  * DKT에서 multi-head attention 처음으로 시도함.
  * exercise embedding을 query로, interaction embedding을 key, value로 projection하여 multi-head attention layer의 input으로 넣음.
  * 이 논문 이후, 후속 모델들이 multi-head attention의 활용이 베이스가 됨.

#### SAINT ([Towards an Appropriate Query, Key and Value Computation for Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3386527.3405945), L@S'20)

<img src='https://user-images.githubusercontent.com/7765506/144063738-0bbf0585-fe00-49a0-baf3-9a9d0c243f8f.jpg' width=600 />

  * DKT에서 Encoder-Decoder 구조를 처음으로 제시함.
  * input을 exercise와 interaction 따로 분리하여 layer를 깊게 쌓을 수 있게 함.

#### AKT ([Context-Aware Attentive Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3394486.3403282), KDD'20)

<img src='https://user-images.githubusercontent.com/7765506/144066136-8b1b3bad-4a87-47b3-b5db-62015a40f333.jpg' width=600 />

  * SAINT와 유사.
  * knowledge retriever embedding이 다시 exercise embedding block의 input 으로 들어감.
  * Motivation: 너무 멀리 떨어진 과거의 interaction은 현재 문제를 푸는 것에 큰 영향을 못 준다. ([망각 곡선](https://ko.wikipedia.org/wiki/망각_곡선) 개념과 유사)
  * 기존 multi-head attention을 변형한 monotonic attention 제안함.
    - 가장 최근의 본 exercise와 interaction의 정보가 많이 반영되도록.

#### 종합

  * 튜닝을 안한 베이스라인 상태에서 4개 모두 비슷한 성능을 보였음.
  * 이후 학습 최적화는 학습 속도가 빠른 SAKT 위주로 진행하였음.

### Traning Optimization

#### Stochastic Weight Averaging (SWA) ([Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407), UAI'18)

  * SGD 특정 step 주기마다 weight을 평균하여 업데이트한다.
  * 모델의 test set에서의 일반화 성능을 높여줄 수 있음.

#### Compute All Seq. Loss & Loss Masking

  * 기존 베이스라인 코드 구현에서 마지막 sequence의 loss만 backprop 시키는데, 이전 history interaction 또한 중요한 정보이기 때문에 모든 sequence의 loss를 backprop 함.
    - 성능 향상이 있었음.
  * 추가적으로 zero padding된 sequence에 대한 정보는 불필요하므로 masking 처리하여 backprop 되지 않도록 함.
    - 추가적인 성능 향상 확인.

#### Random Split Subsequence and Merging

  * 새로운 data sample 생성하기 위해 naive하게 적용해 볼 수 있는 data augmenatation 수행.
    - 주어진 batch를 복사한 후 랜덤 셔플 수행.
    - 기존 batch와 셔플된 batch에서 각각 절반 크기만큼 subsequence를 잘라서 merge 함.
  * 눈에 띄는 성능 향상은 보이지 않으나 특정 하이퍼파라미터에서 같이 사용하면 소폭 성능이 향상되는 듯 보이지만 확실하진 않음.
    - 추후 HPO의 파라미터로 튜닝을 진행했음.
  * 유저 특성이나 유사도를 고려해서 조금 더 그럴듯한 sample을 생성하는 방법에 대한 고민이 필요해 보임.

#### K-Fold Validation

  * 모델 ensemble에 활용함.
  * k=5로 총 5개의 fold에 대해 각각 모델을 학습시킨 후, 각 모델의 prediction 평균함.
  * 성능 향상 확인.

#### Variants of Interaction Embedding

  * Interaction embedding을 구성함에 있어서 앞서 살펴본 논문들에서 (question, answer) 튜플로 구성하는 것이 일반적인 형태인 거 같다.
  * 베이스라인 코드상에서는 (answer,)로만 구성하여 활용하는데 이를 (question, answer)로 교체하였음.
  * 하지만, 오히려 성능하락이 발생하였음.
  * embedding 사이즈를 보면 2 x n_question의 크기르 갖는데, question별 발생 빈도로 보면 user 수에 비해 sparse한 문제가 있어서 그런 거 같음.
  * 그래서 추가적으로 2가지 variant를 적용하여 총 4개의 타입의 interaction에 대해 실험 진행하였음.
    - 0: (answer,)로만 구성하여 incorrect, correct embedding으로만 구성.
    - 1: (question, answer) 튜플로 구성하여 2 x n_question 크기를 갖는 embedding으로 구성.
    - 2: 0번처럼 구성하되, 추가적으로 question embedding과 concat하여 fc layer를 통과시켜 생성. 실질적으로 1번과 같은 space 크기를 갖음.
    - 3: (question, answer) 튜플 대신 (concept, answoer)로 구성. 비교적 concept 수가 상대적으로 적기 때문에 sparse한 문제가 완화됨.
  * 실험결과 특정 방식이 우위라고 할만한 성능 향상은 없었음.
    - 마찬가지로 추후에 HPO의 파라미터로 설정하여 튜닝을 진행하였음.

#### Hyperparameter Optimization (HPO)

  * 끝으로 최적 hyperparmeter를 찾기 위해 SAKT와 AKT 대해 HPO를 수행함.
  * [Optuna](https://optuna.org/)를 사용하여 HPO 진행.
  * 각 모델에 대해 20번씩 trial.
  * SAKT HPO 파라미터
    - `n_heads`: Multi-head attention에서 head 갯수 [2, 4, 8]
    - `interaction_type`: Interaction type [0, 1, 2, 3]
    - `max_seq_len`: Maximum sequence 길이 [20, 40, 60, 80, 100, 200]
    - `hidden_dim`: embedding 및 fc layer hidden dimension [64, 128, 256]
    - `partition_question`: Sequence partition 여부 [true, false]
    - `drop_out`: Dropout ratio [0.2, 0.6] step size 0.1
    - `enable_da`: Data augmentation 여부 [true, false]
  * SAKT HPO 파라미터 search 결과
    - `n_heads`: 4
    - `interaction_type`: 2
    - `max_seq_len`: 20
    - `hidden_dim`: 64
    - `partition_question`: true
    - `drop_out`: 0.4
    - `enable_da`: true
  * AKT HPO 파라미터
    - `n_heads`: Multi-head attention에서 head 갯수 [2, 4, 8]
    - `n_layers`: Multi-head attention 갯수 [2, 4, 8] (모델 아키텍쳐의 `xN`에 해당.)
    - `interaction_type`: Interaction type [0, 1, 2, 3]
    - `max_seq_len`: Maximum sequence 길이 [20, 40, 60, 80, 100, 200]
    - `hidden_dim`: embedding 및 fc layer hidden dimension [64, 128, 256]
    - `final_fc_dim`: Prediction fc layer의 hidden dimension [128, 256, 512]
    - `ff_dim`: 각 block의 fc layer hidden dimension [128, 256, 512, 1024, 2048]
    - `partition_question`: Sequence partition 여부 [true, false]
    - `drop_out`: Dropout ratio [0.2, 0.6] step size 0.1
    - `enable_da`: Data augmentation 여부 [true, false]
  * AKT HPO 파라미터 search 결과
    - `n_heads`: 4
    - `n_layers`: 4
    - `interaction_type`: 2
    - `max_seq_len`: 200
    - `hidden_dim`: 128
    - `final_fc_dim`: 512
    - `ff_dim`: 128
    - `partition_question`: true
    - `drop_out`: 0.3
    - `enable_da`: false

#### Insight

  * 두 모델 모두 `interaction_type`: 2로 선택하였다.
    - interation embedding space가 어느 정도 크면 좋은 듯 보임.
    - 단, 1번 옵션과 같이 navive하게 구성하는 것보다 2번과 같이 다른 information과 결합하는 형태가 좋아보임.
  * 두 모델 모두 `partition_question`을 사용하였다.
    - Sequence를 잘라내는 것 보다 재사용하는 것이 효과적인 듯 하다.
  * `max_seq_len` & `enable_da` (가설)
    - SAKT는 20, false를, AKT는 200, true를 선택하였다.
    - 짧은 sequence인 경우, data augmentaion 효과가 없어 보임.
    - AKT는 contribution에 언급한 바와 같이 긴 sequence에서도 잘 동작하는 듯 보임. (monotonic attention)
    - 긴 sequence인 경우 zero padding이 많아지기 때문에 data augmentation 효과가 있는 듯 하다.

## Experimental Results

### Baseline 성능

  * 튜닝이 덜된 베이스라인 상태에서 대부분의 모델 아키텍쳐들의 성능이 비슷함.
  * AKT의 경우, 학습 최적화를 어느 정도 진행하고 돌려서 성능이 높게 나옴.
    - 따라서 AKT가 좋다고 판단할 수는 없음.

| Model | AUC    | ACC    |
|-------|--------|--------|
| LSTM  | 0.7280 | 0.6720 |
| SAKT  | 0.7120 | 0.6590 |
| SAINT | 0.7180 | 0.6400 |
| AKT   | 0.7690 | 0.6770 |

### Dataset Partition 효과

기존 베이스라인 코드는 `max_seq_len`보다 길면 나머지 sequence를 잘라서 버리는데, 데이터의 소실을 줄이기 위해 버리지 않고 새로운 sequence로 구성하여 기존 데이터셋에 포함. 일종의 data augmentation 효과.

| Model            | AUC    | ACC    |
|------------------|--------|--------|
| LSTM             | 0.7280 | 0.6720 |
| LSTM + partition | 0.7420 | 0.6830 |


### K-Fold 효과

k = 5로 5개의 fold를 만들어서 각 모델을 training한 후에 각 모델의 prediction을 average.

| Model         | AUC    | ACC    |
|---------------|--------|--------|
| SAKT          | 0.7490 | 0.6800 |
| SAKT + K-Fold | 0.7780 | 0.6990 |

### Compute All Seq. Loss & Loss Masking 효과

| Model                       | AUC    | ACC    |
|-----------------------------|--------|--------|
| SAKT                        | 0.7780 | 0.6990 |
| SAKT + Modified Loss Scheme | 0.7980 | 0.7310 |

### HPO 결과 (최종)

  * AUC는 향상되었는데 오히려 ACC가 떨어짐.
  * Search trial를 늘리거나 ACC에 대해서도 objective를 추가하는 것이 필요해 보임.

| Model      | AUC    | ACC    |
|------------|--------|--------|
| SAKT       | 0.7980 | 0.7310 |
| SAKT + HPO | 0.7995 | 0.6989 |

### 기타

  * interaction 만드는 과정에서 interaction에 대한 mask도 shilft해서 적용해야 하는데 그러지 않은 것을 발견.
  * 수정하여 적용해보지 못했으나 수정하면 성능향상이 있을 것으로 보임.
  * AKT에 대해 위의 문제를 수정 후 HPO 파라미터로 학습시에 validataion에 대한 성능이 0.8180 정도까지 오르는 것을 확인.

## Future Work

  * Interaction embedding 생성에 대한 다른 접근법에 대한 필요성.
  * Semantic을 고려할 수 있는 data augmentation 전략.
  * 문제의 텍스트 등과 같이 embedding을 조금 더 잘 만들 수 있는 side information의 활용.
  * Pre-training, Fine-tuning
    - NLP, computer vision에서 효과가 검증된 pre-training 기법 적용.
    - 다양한 dataset으로부터 pre-trained model 학습 후 특정 task에 fine-tuning하는 전략.
    - [Cross-Market Product Recommendation](https://dl.acm.org/doi/abs/10.1145/3459637.3482493), CIKM'21

## Envrionment

  * python: 3.7.x
  * cuda: 11.x
  * gpu: RTX2080Ti x 2 | RTX8000 x 2

### requirements.txt

```
torch==1.10.0
pandas==1.3.4
sklearn==0.0
tqdm==4.62.3
wandb==0.12.7
transformers==4.12.5
logzero==1.7.0
optuna==2.10.0
click==8.0.3
ruamel.yaml==0.17.17
attrdict==2.0.1
```

### pip

```bash
$ cd [project]  # project root 폴더
$ pip install -r requirements.txt  # python package 설치
```

### Docker

[pip](#pip) 대신 Docker 이용할 경우.

  * cuda >= 11.0
  * [Docker](https://www.docker.com) >= 19.03.5
  * [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```bash
$ cd [project]  # project root 폴더
$ docker build -t [image_name] .  # Docker 이미지 빌드
$ docker run --rm -t -d --name [container_name] --gpus all -v $PWD:/workspace -w /workspace \
--ipc=host --net=host [...options] [image_name]  # docker container 생성
$ docker exec -it [container_name] zsh  # docker container shell 진입
```

## Instructions

### Reproduce

```bash
$ cd [project]
$ ./scripts/run_hptuning.sh  # hpo (optional)
$ ./scripts/run_train.sh  # 모델 학습 및 inference 수행
```
  
  * 데이터 파일은 `data/{train|test}_data.csv`에 추가한다. 
  * 실행 후 결과 파일은 `output/[YYYYMMDD_HHMMSS]_hptuning`, `output/[YYYYMMDD_HHMMSS]`에 생성된다.

### Training Model

```bash
$ python train.py --help


usage: train.py [-h] [--seed SEED] [--device DEVICE] [--data_dir DATA_DIR]
                [--asset_dir ASSET_DIR] [--file_name FILE_NAME]
                [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                [--output_root_dir OUTPUT_ROOT_DIR] [--output_dir OUTPUT_DIR]
                [--output_filename OUTPUT_FILENAME]
                [--test_file_name TEST_FILE_NAME] [--max_seq_len MAX_SEQ_LEN]
                [--num_workers NUM_WORKERS] [--partition_question]
                [--interaction_type INTERACTION_TYPE] [--random_permute]
                [--hidden_dim HIDDEN_DIM] [--n_layers N_LAYERS]
                [--n_heads N_HEADS] [--drop_out DROP_OUT]
                [--attn_direction ATTN_DIRECTION] [--l2 L2]
                [--final_fc_dim FINAL_FC_DIM] [--ff_dim FF_DIM]
                [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                [--clip_grad CLIP_GRAD] [--patience PATIENCE]
                [--swa-warmup SWA_WARMUP] [--compute_loss_only_last]
                [--k_folds K_FOLDS] [--inference_only] [--log_steps LOG_STEPS]
                [--enable_da] [--model MODEL] [--optimizer OPTIMIZER]
                [--scheduler SCHEDULER]
```

### HPO

```bash
$ python hptuning.py --help


Usage: hptuning.py [OPTIONS]

Options:
  --model [lstm|sakt|saint|akt]   model  [default: sakt]
  --output-root-dir PATH          output root directory
  --output-dir PATH               Set output directory
  --config-file-path PATH         hp params config file path  [default:
                                  config/hp_params.yaml]
  --default-param-file-path PATH  default param file path  [default:
                                  config/default_args.json]
  --seed INTEGER                  seed  [default: 42]
  --n-trials INTEGER              # of trials  [default: 20]
  --study-name TEXT               Set study name  [default: study]
  --storage-name TEXT             Set storage name to save study  [default:
                                  storage]
  --n-epochs INTEGER              # of epochs  [default: 100]
  --patience INTEGER              early stop patience  [default: 8]
  --clip-grad FLOAT               clipping value  [default: 5.0]
  --lr FLOAT                      learning rate  [default: 0.001]
  --batch-size INTEGER            batch size  [default: 128]
  --model-name TEXT               model name  [default: model.pt]
  --logfile-name TEXT             logfile name  [default: train.log]
  --help                          Show this message and exit.  [default:
                                  False]
```

## References

  * Shalini Pandey, George Karypis. [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837.pdf). In EDM, 2019.
  * Choi, Youngduck and Lee, Youngnam and Cho, Junghyun and Baek, Jineon and Kim, Byungsoo and Cha, Yeongmin and Shin, Dongmin and Bae, Chan and Heo, Jaewe. [Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3386527.3405945). In L@S, 2020.
  * Ghosh, Aritra and Heffernan, Neil and Lan, Andrew S. [Context-Aware Attentive Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3394486.3403282). In KDD, 2020.
  * Pavel Izmailov and Dmitrii Podoprikhin and Timur Garipov and Dmitry Vetrov and Andrew Gordon Wilson. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). In UAI, 2018.
  * Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James. [Cross-Market Product Recommendation](https://dl.acm.org/doi/abs/10.1145/3459637.3482493). In CIKM, 2021.
  * Takuya Akiba and Shotaro Sano and Toshihiko Yanase and Takeru Ohta and Masanori Koyama. [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902). In KDD, 2019.
