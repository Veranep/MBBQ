# MBBQ
This is the repository for the Multilingual Bias Benchmark for Question-answering (MBBQ) dataset.

Authors: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández.

## The paper
This dataset was introduced in [MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs](https://openreview.net/forum?id=X9yV4lFHt4#discussion).
![Dataset](MBBQ_dataset.png)

## About MBBQ (paper abstract)
Generative large language models (LLMs) have been shown to exhibit
harmful biases and stereotypes. While safety fine-tuning typically takes
place in English, if at all, these models are being used by speakers of many
different languages. There is existing evidence that the performance of these
models is inconsistent across languages and that they discriminate based on
demographic factors of the user. Motivated by this, we investigate whether
the social stereotypes exhibited by LLMs differ as a function of the language
used to prompt them, while controlling for cultural differences and task
accuracy. To this end, we present MBBQ (Multilingual Bias Benchmark for
Question-answering), a carefully curated version of the English BBQ dataset
extended to Dutch, Spanish, and Turkish, which measures stereotypes
commonly held across these languages. We further complement MBBQ
with a parallel control dataset to measure task performance on the question-
answering task independently of bias. Our results based on several open-
source and proprietary LLMs confirm that some non-English languages
suffer from bias more than English, even when controlling for cultural shifts.
Moreover, we observe significant cross-lingual differences in bias behaviour
for all except the most accurate models. With the release of MBBQ, we hope
to encourage further research on bias in multilingual settings.

## Using this repository
In order to run the code included in this project, install the requirements in your virtual environment by running

```
pip install -r requirements.txt
```

- The `data` folder contains all samples from MBBQ, separated by subset, language, and control set.
- `mbbq.py` contains the code to embed the samples in the 5 prompts, prompt the models, and detect the answer from their response:
```
python mbbq.py -mode generate_samples -subsets Age Disability_status Gender_identity Physical_appearance SES Sexual_orientation -lang nl -model CohereForAI/aya-101 -exp_id aya_nl_control --control
python mbbq.py -mode ask_model -subsets Age Disability_status Gender_identity Physical_appearance SES Sexual_orientation -lang nl -model CohereForAI/aya-101 -bs 16 -exp_id aya_nl_control --control
python mbbq.py -mode detect_answers -subsets Age Disability_status Gender_identity Physical_appearance SES Sexual_orientation -lang nl -model CohereForAI/aya-101 -exp_id aya_nl_control --control
```
- `models.py` contains the code to load the models and generate responses.
- `answer_detection.py` contains the prompts, and phrases that are used for detecting answers in the model responses.

**Note:** MBBQ is intended for model evaluation and should *NOT* be used for model training. The bias scores obtained from evaluation on MBBQ are an indication of the social biases present in a model, but are no guarantee for the model's behavior in other settings.

## Citation
If you use the code in this repository, please cite the following paper:
```
@inproceedings{
neplenbroek2024mbbq,
title={{MBBQ}: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative {LLM}s},
author={Vera Neplenbroek and Arianna Bisazza and Raquel Fern{\'a}ndez},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=X9yV4lFHt4}
}
```
