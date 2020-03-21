# CNN-BiLSTM-CRF-NER

Korean NER Task with CharCNN + BiLSTM + CRF (with Naver NLP Challenge dataset)

## Model

<p float="left" align="center">
    <img width="400" src="https://user-images.githubusercontent.com/28896432/77224229-d9bce580-6ba6-11ea-9564-06d57a2e0f09.png" />  
</p>

- Character Embedding with `CNN`
- Concatenate `word embedding` with `character represention`
- Put the feature above to `BiLSTM + CRF`

## Dependencies

- python>=3.5
- torch==1.4.0
- seqeval==0.0.12
- pytorch-crf==0.7.2
- gdown==3.10.1

## Data

|           | Train  | Test  |
| --------- | ------ | ----- |
| # of Data | 81,000 | 9,000 |

- Naver NLP Challenge 2018 NER Dataset ([Github link](https://github.com/naver/nlp-challenge))
- Original github only has train dataset, so test dataset is created by splitting the train dataset. ([Data link](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging/data))

## Pretrained Word Vectors

- Use [Korean fasttext vectors](https://fasttext.cc/docs/en/crawl-vectors.html) with 300 dimension
- It takes quiet long time to load from original vector, so I take out the word vectors that are only in word vocab.
- **It will be downloaded automatically when you run `main.py`.**

## Usage

```bash
$ python3 main.py --do_train --do_eval
```

- **Evaluation prediction result** will be saved in `preds` dir when you give `--write_pred` option.

## Results

|                | Slot F1 (%) |
| -------------- | ----------- |
| CNN+BiLSTM+CRF | 74.57       |

## Reference

- [Naver NLP Challenge](https://github.com/naver/nlp-challenge)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)
- [NLP Implementation by aisolab](https://github.com/aisolab/nlp_implementation)
