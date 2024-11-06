# README-demo

# ‘Pseudo-labeling with keyword refining for few-supervised video captioning’

This is official pytorch implementation of Pseudo-labeling with keyword refining for few-supervised video captioning. The pdf can be downloaded from here

## **Abstract**

Video captioning aims to generate one sentence that describes the video content. Existing methods always require a number of captions (e.g.10 or 20) per video to train the model, which is quite costly due to the expensive human labeling. To reduce the labeling cost, we explore the possibility of using only one ground-truth sentences to train the model, and introduce a new task named few-supervised video captioning To fulfil the task, we strive to generate high-quality pseudo-labeled sentences to augment supervision information, such that the generalization ability of the model is improved. In particular, we propose one few-supervised video captioning framework that expands supervision knowledge by generating promising pseudo labels.Unlike random sampling in natural language processing that may cause invalid modifications (i.e.,edit words), we adopt a two-step strategy in lexically constrained sentence generation, which guides the model to edit words using some actions (e.g., copy, substitute,delete,and insert)by pre-trained label classifier, and fine-tunes generated sentences by pre-trained language model. Moreover,to minimize the inconsistency between pseudo-labeled sentences and video content, we select sentences by video-text retrieval,and design transformer-based caption generation module with gated video-keyword semantic fusion,which filters out irrelevant words in semantic cues and infers relevant words by visual cues.In addition,to reduce word repetitions,we adopt a repetitive penalized sampling method to encourage the model to yield concise pseudo-labeled sentences with less repeated words. Extensive experiments on several benchmarks have demonstrated the advantages of the proposed approach for few-supervised video captioning.

## **Requirements**

The main runtime environment of the code:

```python
 python 3.8
 Pytorch 1.8.0
 CUDA 10.2
```

Install the libraries using requirements.txt as:

```python
 pip install -r requirements.txt
```

## **Dataset**

For training, download the VATEX dataset into the ‘video_captioning/datasets/vatex/’ directory, baidu pan: **[download link](https://pan.baidu.com/s/1aFP6B6FgGTomMQ6iWM_U_g?pwd=mo2c), Extracted code: m02c**. This includes video features, video annotation on fully supervised, and video annotation on few supervised 

```python
├──video_captioning
		├──datasets
		   ├──vatex
		      ├──features
		         ├──Faster-rcnn
		            ├──1.npy
                            ├──2.npy
                            ├──*.npy
		         ├──InceptionResnetv2
                            ├──1.npy
                            ├──2.npy
                            ├──*.npy
                         ├──Resnext
                            ├──1.npy
                            ├──2.npy
                            ├──*.npy
                      ├──full_supervision
                         ├──captions_test.json
                         ├──captions_train.json
                         ├──captions_val.json
                         ├──dataset.json
                         ├──vatex_caption_anno_test.pkl
                         ├──vatex_caption_anno_train.pkl
                         ├──vatex_caption_anno_val.pkl
                         ├──vocabulary.txt
                      ├──few_supervision
                         ├──captions_test.json
                         ├──captions_train.json
                         ├──captions_val.json
                         ├──dataset.json
                         ├──vatex_caption_anno_test.pkl
                         ├──vatex_caption_anno_train.pkl
                         ├──vatex_caption_anno_val.pkl
		         ├──vocabulary.txt
```

## Experimental settings

For the VATEX dataset, set the maximum keyword length to 7. The number of appearance and motion feature vectors is consistent, with a unified number of 30 in the Vatex dataset. Similarly, the number of target vectors in each video is also unified, with a unified number of 30 in the Vatex dataset. 

For captions, first remove punctuation and convert all letters to lowercase, remove punctuation, and include a start tag "<BOS>" and an end tag "<EOS>" in each caption. For specific experimental settings, refer to the yaml file in "/video_capting/configurations/video_caption/vatex”

## **Training & Testing**

Our code is mainly divided into three parts: pseudo label generation and video Captioning. Next, we will introduce the training and testing methods for these two parts in sequence. If you have downloaded the above dataset, you can skip 1 and run video captioning directly

### **1. Pseudo label generation**

We first use COCO caption to create synthetic data, and then fine-tune XLNet (base-cased version) on them to get the token-level classifier. Next, we train forward and backward language models, and use them as the candidate generator. Finally, we refine the candidate sentence with the classifier and repetitive penalized sampling.

- Pre-processing: Tokenize the raw text with XLNet (based-cased) tokenizer. Make sure that the directory of the dataset (e.g., "data/VATEX") is empty. Then, you should prepare some sentences to construct the training set (one sentence in each line). This file is named as "train.txt". Similarly, you should prepare some sentences to construct the validation set, which is named as "validation.txt" in our propgram. You should put 'train.txt' and 'validation.txt' in the correspoinding dataset directory (e.g., "data/VATEX"). You should prepare keywords to constrct the test set mentioned in the paper. Please refer to for details.
    
    ```python
    cd language_models
    python Xlnet_MaskedLM.py --convert_data 1
    ```
    
- Step 1: fine-tune XLNet on the masked lm dataset
    
    ```python
    nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 Xlnet_MaskedLM.py --gpu 0,1,2,3 --dataset VATEX >vatex_Xlnet_MaskedLM.log &
    ```
    
- Step 2: create synthetic data for training the XLNet-based classifier
    
    ```python
    cd utils
    nohup python create_synthetic_data.py --generate_mode 2 --batch_size 100 --train_dataset_size -1 --validation_dataset_size -1 --dataset VATEX >vatex_create_synthetic_data.log &
    ```
    
- Step 3: train the XLNet-based classifier
    
    ```python
    cd classifier
    nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 29504 xlnet_classifier.py --gpu 0,1,2,3 --dataset VATEX >vatex_xlnet_classifier.log &
    ```
    
- Step 4: train language models (Note: You can train all language models at the same time. )
    
    • Train the forward XLNet-based language model
    
    ```python
    cd language_models
    nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 Xlnet_LM.py --gpu 2 --is_forward 1 --dataset VATEX --train 1 >vatex_xlnet_forward.log &
    ```
    
    • Train the backward XLNet-based language model
    
    ```python
    cd language_models
    nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 Xlnet_LM.py --gpu 3 --is_forward 0 --dataset VATEX --train 1 >vatex_xlnet_backward.log &
    ```
    
    **You can directly use the model we provide to skip the above steps, you should download VATEX checkpoints, then put them into the 'checkcpoints' directory.**
    
- Step 5: generete with XLNet-based MCMC model (X-MCMC)

```python
nohup python main.py --model_name XLNetLMGenerate --random 0 --gpu 0 --max_keywords 6 -sn 50 --split 4000 --dataset VATEX --sentence_nums 5 >VATEX_keywords_6_generate_split_4000_nums_5.log &
```

**We also provide the pseudo tags generated on the VATEX dataset, then put them into the '/outputs/XLNetLMGenerate/VATEX' directory.**

### 2. Video-text retrieval

- Step 1: obtain xclip features of the video on the VATEX dataset, You can also download it directly

```python
cd video-text_retrieval
nohup python vidfeats.py --videos_path the_path_where_you_saved_the_video --save_path 
the_path_where_you_saved_the_video_features
```

- Step 2: obtain pseudo labels that are most relevant to video content

```python
nohup python main.py --sents_top_k select_the_top-k_sentences
```

## 3. Video captioning

### 3.1 Full supervision

Under fully supervised conditions, there is no need for the above two steps, just run the video captioning directly. 

**Training the VATEX:**

```python
cd video_captioning
CUDA_VISIBLE_DEVICES=0,1 nohup python3 train_net.py --config-file /configs/video_caption/vatex/full_supervision.yaml --dist-url tcp://127.0.0.1:5000 --num-gpus 2 OUTPUT_DIR /output/vatex/full_supervision DATALOADER.TRAIN_BATCH_SIZE 128 DECODE_STRATEGY.BEAM_SIZE 5 DATALOADER.NUM_WORKERS 4 >log_vatex_full_supervision.log &
```

**Testing the VATEX:**

Before testing, place the trained model file in the ‘/checkpoint/vatex/full_supervision’ directory, You can also use the model we provide, then use the command below for testing. In addition, you need to download ‘coco-captions’ and place them in ‘/video_captioning/xmodaler’:

```python
CUDA_VISIBLE_DEVICES=0 nohup python3 train_net.py --config-file /configs/video_caption/vatex/full_supervision.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS /checkpoint/vatex/full_supervision/model_final.pth
```

### 3.2 Few supervision

**Training the VATEX:**

```python
CUDA_VISIBLE_DEVICES=0,1 nohup python3 train_net.py --config-file /configs/video_caption/vatex/few_supervision.yaml --dist-url tcp://127.0.0.1:5000 --num-gpus 2 OUTPUT_DIR /output/vatex/few_supervision DATALOADER.TRAIN_BATCH_SIZE 128 DECODE_STRATEGY.BEAM_SIZE 5 DATALOADER.NUM_WORKERS 4 >log_vatex_few_supervision.log &
```

**Testing the VATEX:**

Before testing, place the trained model file in the ‘checkpoint/vatex/few_supervision’ directory, You can also use the model we provide, then use the command below for testing:

```python
CUDA_VISIBLE_DEVICES=0 nohup python3 train_net.py --config-file /configs/video_caption/vatex/few_supervision.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS /checkpoint/vatex/few_supervision/model_final.pth
```

## Results

### 1. Quantitative results on VATEX

**1.1 Full supervision**

Comparison of methods under fully supervised conditions, all models use 10 ground truth

| Model | Venue | Blue@4 | METEOR | ROUGE-L | CIDEr-D |
| --- | --- | --- | --- | --- | --- |
| MAN | TMM’23 | 32.7 | 22.4 | 49.1 | 48.9 |
| Ours |  | 34.2(+1.6) | 23.5(+1.1) | 49.5(+0.4) | 53.3(+4.4) |

**1.2 Few supervision**

Comparison of methods under fully supervised conditions, all models use one ground truth and one pseudo label

| Model | Venue | Blue@4 | METEOR | ROUGE-L | CIDEr-D |
| --- | --- | --- | --- | --- | --- |
| MAN | TMM’23 | 26.6 | 17.4 | 43.2 | 37.1 |
| Ours |  | 27.6(+1.3) | 18.4(+1.0) | 44.1(+0.9) | 40.4(+3.3) |

The Few supervision provides one ground truth and one pseudo labels.

### 2. Qualitative results on VATEX

<p align="center">
<img src="qualitative_results_vatex.png" width="800"> <br>
</p>
'Missing' represents words and text fragments that the model did not predict，(a)HMN (b)HRN

## **Acknowledgement:**

The implement of this project is based on the codebases bellow.

[MCMCXLNet](https://github.com/NLPCode/MCMCXLNet)

[xmodal](https://github.com/yehli/xmodaler)

If you find this project helpful, Please also cite codebases above.

## Citation

If you find this repo useful, please cite the following paper.
```
@article{li2025vidcaption,
      title={Adversarial Attacks on Video Object Segmentation with Hard Region Discovery}, 
      author={Ping Li and Tao Wang and Xinkui Zhao and Xianghua Xu and Mingli Song},
      year={2025},
      journal ={2309.13857},
      year = {2025},
      doi = {}
}
```
## Contact:

Please drop me an email for any problems or discussion: wangtao000213@hdu.edu.cn
