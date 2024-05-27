# ZS_ner_dip
Эта работа основана на CROP Zero-shot Cross-lingual Named Entity Recognition with Multilingual Labeled Sequence Translation с некоторыми модификациями


## Данные

Использовался набор данных **XTREME-40**.


## Окружение 

* Python: >= 3.7
* [PyTorch](http://pytorch.org/): >= 1.5.0
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* [apex](https://github.com/NVIDIA/apex): >= 0.1 ТОЛЬКО ДЛЯ ОБУЧЕНИЯ
* [Fairseq](https://github.com/pytorch/fairseq)*: 1.0.0

\* - устанавливается из папки m2m командой pip install --editable ./

## Использование пайплайна

### Обучение моделей для NER и для перевода

**Исходная модель NER**

```bash
bash ./pipeline/step0_train_source_ner_model.sh
```

**Модель перевода многоязычной последовательности с маркировкой**

```bash
bash ./pipeline/step0_train_translation_model.sh
```

* Скачать: [Google Drive](https://drive.google.com/drive/folders/1dfrgOmMIrmphbYQkfyH5K_iOOtqC9k8Q?usp=sharing);
  * Предобученные модели
    * Обученная базовая модель перевода: `m2m_checkpoint_baseline.pt`
    * Обученная модель перевода на основе вставок: `m2m_checkpoint_insert_avg_41_60.pt`
    * Обученная модель перевода, основанная на замене: `m2m_checkpoint_replace_avg_11_20.pt`
  * Словарь для токенизации (используется всеми тремя описанными выше моделями): `dict.txt`
    * `dict-40-lang.zip` включает в себя 40 словарей на разных языках.
  * Модель фрагментирования слов: `spm.model`
  * XTREME-40 набор данных для NER: `xtreme_ner_data.zip`

### Основной пайплайн

1. Подготовка данных целевого языка

```bash
bash ./pipeline/step1_prepare_tgt_translation_data.sh
```

2. Перевод данных целевого языка на язык источник

(использована модель flores101_mm100_615M)

```bash
bash ./pipeline/step2_tgt2src_translation.sh
```

3. Подготовка переведенных данных для NER 

```bash
bash ./pipeline/step3_preapre_src_ner_data.sh
```

4. Распознавание на языке источнике

```bash
bash ./pipeline/step4_src_ner.sh
```

5. Подготовка данных для перевода 

```bash
bash ./pipeline/step5_prepare_src_translation_data.sh
```

6. Маркированный перевод на целевой язык

(Для воспроизведение необходимо использовать m2m_checkpoint_insert_avg_41_60.pt или иную обученную использовать проекцию)

```bash
bash ./pipeline/step6_labeled_transation.sh
```

7. Подготовка и фильтрация полученных данных

```bash
bash ./pipeline/step7_prepare_pseudo_ner_data1.sh
bash ./pipeline/step7_prepare_pseudo_ner_data2.sh
```
