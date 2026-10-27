---
annotations_creators:
- crowdsourced
- expert-generated
language_creators:
- crowdsourced
- expert-generated
language:
- amh
- arb
- ary
- ars
- acq
- arz
- apc
- ben
- ceb
- dan
- deu
- ell
- eng
- eus
- fil
- fin
- fra
- gle
- guj
- hat
- hau
- hin
- hun
- ibo
- ind
- ita
- jav
- jpn
- kan
- kir
- kor
- kur
- lit
- mal
- mar
- mlg
- msa
- mya
- nep
- nld
- nso
- nya
- pan
- pes
- pol
- por
- pus
- rus
- sin
- sna
- snd
- som
- spa
- sqi
- srp
- sun
- swa
- swe
- tam
- tel
- tha
- tur
- ukr
- urd
- vie
- wol
- xho
- yor
- zho
- zul
license: apache-2.0
multilinguality:
- multilingual
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- other
task_ids: []
pretty_name: Aya Dataset
dataset_info:
- config_name: default
  features:
  - name: inputs
    dtype: string
  - name: targets
    dtype: string
  - name: language
    dtype: string
  - name: language_code
    dtype: string
  - name: annotation_type
    dtype: string
  - name: user_id
    dtype: string
  splits:
  - name: test
    num_bytes: 1782208
    num_examples: 1750
  - name: train
    num_bytes: 254591851
    num_examples: 202362
  download_size: 275359572
  dataset_size: 256374059
- config_name: demographics
  features:
  - name: user_id
    dtype: string
  - name: age_range
    sequence: int64
  - name: gender
    dtype: string
  - name: country
    dtype: string
  - name: languages
    sequence: string
  - name: dialects
    sequence: string
  splits:
  - name: train
    num_bytes: 202127
    num_examples: 1456
  download_size: 113702
  dataset_size: 202127
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
- config_name: demographics
  data_files:
  - split: train
    path: demographics/train-*
tags: []
---
![Aya Header](https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/aya_header.png)
# Dataset Summary
The `Aya Dataset` is a multilingual instruction fine-tuning dataset curated by an open-science community via [Aya Annotation Platform](https://aya.for.ai/) from Cohere Labs. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators.<br>
This dataset can be used to train, finetune, and evaluate multilingual LLMs.

- **Curated by:** Contributors of [Aya Open Science Intiative](https://aya.for.ai/).
- **Language(s):** 65 languages (71 including dialects & scripts).
- **License:** [Apache 2.0](https://opensource.org/license/apache-2-0)
- **Aya Datasets Family:**

  | Name | Explanation |
  |------|--------------|
  | [aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-annotated multilingual instruction finetuning dataset, comprising over 204K instances across 65 languages. |
  | [aya_collection](https://huggingface.co/datasets/CohereLabs/aya_collection) | Created by applying instruction-style templates from fluent speakers to 44 datasets, including translations of 19 instruction-style datasets into 101 languages, providing 513M instances for various tasks.|
  | [aya_collection_language_split](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Aya Collection structured based on language level subsets. |
  | [aya_evaluation_suite](https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite) | A diverse evaluation set for multilingual open-ended generation, featuring 250 culturally grounded prompts in 7 languages, 200 translated prompts in 24 languages, and human-edited versions selected for cross-cultural relevance from English Dolly in 6 languages.|
  | [aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming)|  A red-teaming dataset consisting of harmful prompts in 8 languages across 9 different categories of harm with explicit labels for "global" and "local" harm.|

# Dataset
The `Aya Dataset` comprises of two types of data:
1. **Human Annotations:** Original annotations (brand new prompts and completions written by annotators) and re-annotations (human edits of automatically generated prompts and completions). 
2. **Demographics Data:** Anonymized information for each annotator. 

## Load with Datasets
To load this dataset consisting of both prompt-completions and demographics data with `datasets`, you'll just need to install Datasets as `pip install datasets --upgrade` and then use the following code:

```python
from datasets import load_dataset

# Load the annotations dataset
aya_dataset = load_dataset("CohereLabs/aya_dataset")

# Load the demographics dataset
aya_demographics = load_dataset("CohereLabs/aya_dataset", "demographics")
```

## Data Fields
### Human Annotations (Default)
The data fields are the same among all splits:
- `inputs`: Prompt or input to the language model.
- `targets`: Completion or output of the language model.
- `language`: The language of the `inputs` and `targets`.
- `language_code`: The ISO code for the language of the `inputs` and `targets`.
- `annotation_type`: The value denoting whether `inputs` and `targets` are 'original_annotations' or 're-annotations'.
- `user_id`: Unique identifier of the annotator who submitted the prompt-completion pair.

### Demographics Data
The data fields are the same among all splits:
- `user_id`: Unique identifier of the annotator who submitted the prompt-completion pair.
- `age_range`: Age of the annotator. Ranges from 0 to 121.
- `gender`: Gender of the annotator. The values are 'male', 'female', 'prefer not to say', 'non-binary' and 'others'.
- `languages`: List of languages spoken by the annotator.
- `dialects`: Dialects reported by the annotator.

Some empty values may be represented as 'null'.

## Data Splits
### Human Annotations (Default)
The following are the splits of the data:
| Split | No. of instances | Language Coverage |
|-------|------------------|-------------------|
| train | 202,364 | All |
| test  | 1,750 | 7 ('Standard Arabic', 'Yoruba', 'Turkish', 'English', 'Simplified Chinese', 'Portuguese', 'Telugu')|

### Demographics Data
The following are the splits of the data:
| Split | No. of Instances |
|-------|------------------|
| train | 1,456 |

## Data Instances
### Human Annotations (Default)
An example of `train` looks as follows:
```json
{
"inputs": "What cultural events or festivals add vibrancy to Colombo's calendar...",
"targets": "Colombo's cultural calendar is adorned with diverse events and festivals that celebrate the city's rich tapestry of traditions...",
"language": "English",
"language_code": "eng",
"annotation_type": "original-annotations",
"user_id": "f0ff69570af705b75c5a0851883e..."
}
```

### Demographics Data
An example of `train` looks as follows:
```json
{
"user_id": "f0ff69570af705b75c5a0851883e...",
"age_range": [ 25, 35 ],
"gender": "female",
"languages": [ "English", "Hausa" ],
"dialects": [ "Hausa" ]
}
```

## Statistics
### Annotation Types
The following is the breakdown of original annotations and re-annotations in the final dataset.

| Type of Annotation | Instances |
|--------------------|-----------|
| Original Annotations | 138,844 |
| Re-Annotations | 65,270 |
| Total | 204,114|

### Languages
The dataset covers 65 languages: 28 high-resource, 12 mid-resource, and 31 low-resource languages. The following is details about the languages, dialects & scripts included in the dataset.

<details>
<summary> Languages Info </summary>

| ISO Code | Language | Resources |
|----------|----------|-----------|
| `amh` | Amharic | Low |
| `arb`, `ary`, `ars`, `acq`, `arz` & `apc`  | Arabic (Standard, Moroccan, Najdi, Ta'izzi-Adeni, Egyptian & South Levantine) | High |
| `ben` | Bengali | Mid |
| `ceb` | Cebuano | Mid |
| `dan` | Danish | Mid |
| `deu` | German | High |
| `ell` | Greek | Mid |
| `eng` | English | High |
| `eus` | Basque | High |
| `fil` | Filipino | Mid |
| `fin` | Finnish | Mid |
| `fra` | French | High |
| `gle` | Irish | Low |
| `guj` | Gujarati | Low |
| `hat` | Haitian Creole | Low |
| `hau` | Hausa | Low |
| `hin` | Hindi | High |
| `hun` | Hungarian | High |
| `ibo` | Igbo | Low |
| `ind` | Indonesian | Mid |
| `ita` | Italian | High |
| `jav` | Javanese | Low |
| `jpn` | Japanese | High |
| `kan` | Kannada | Low |
| `kir` | Kyrgyz | Low |
| `kor` | Korean | Mid |
| `kur` | Kurdish | Low |
| `lit` | Lithuanian | Mid |
| `mal` | Malayalam | Low |
| `mar` | Marathi | Low |
| `mlg` | Malagasy | Low |
| `msa` | Malay | Mid |
| `mya` | Burmese | Low |
| `nep` | Nepali | Low |
| `nld` | Dutch | High |
| `nso` | Northern Sotho | Low |
| `nya` | Chichewa | Low |
| `pan` | Punjabi | Low |
| `pes` | Persian | High |
| `pol` | Polish | High |
| `por` | Portuguese | High |
| `pus` | Pashto | Low |
| `rus` | Russian | High |
| `sin` | Sinhala | Low |
| `sna` | Shona | Low |
| `snd` | Sindhi | Low |
| `som` | Somali | Low |
| `spa` | Spanish | High |
| `sqi` | Albanian | Low |
| `srp` | Serbian | High |
| `sun` | Sundanese | Low |
| `swa` | Swahili | Low |
| `swe` | Swedish | High |
| `tam` | Tamil | Mid |
| `tel` | Telugu | Low |
| `tha` | Thai | Mid |
| `tur` | Turkish | High |
| `ukr` | Ukrainian | Mid |
| `urd` | Urdu | Mid |
| `vie` | Vietnamese | High |
| `wol` | Wolof | Low |
| `xho` | Xhosa | Low |
| `yor` | Yorùbá | Low |
| `zho` | Chinese (Traditional & Simplified) | High |
| `zul` | Zulu | Low |
</details>
<br>

# Motivations & Intentions
- **Curation Rationale:** The curation effort employed an open-science approach to create a diverse instruction-style dataset through annotators across the globe that ensures comprehensive representation across all languages. The success of the curation effort, led by volunteers across diverse backgrounds, was significantly influenced by their hope to meaningfully bring NLP advancements to their languages.


# Known Limitations
- **Language and dialect coverage:** The dataset covers a limited fraction of the world's linguistic diversity, with 93% of languages not represented, facing challenges in distinguishing between languages and dialects, lacking coverage for many regional dialects, and excluding programming languages.
- **Uneven distribution of contributions:** The dataset contains contributions in annotation activities, with a 'long tail' of annotators making only one or two contributions, leading to potential dataset imbalances across languages and a lack of diversity within certain language annotations.
- **Cultural and Personal Bias:** In the dataset, certain languages have limited representation due to a few dominant annotators, potentially leading to a narrow viewpoint and skewed distribution of content, particularly towards certain domains like news.
- **Gendered Pronouns:** Many of the languages in the Aya Dataset only contain pronouns that are explicitly gendered (e.g., Arabic) or that lack gender-neutral third-person pronouns for gender-neutral reference (e.g. Estonian).
- **Formality Distinctions:** The dataset encompasses languages with diverse formality distinctions, involving honorifics and situational choices in pronoun use, reflecting varying levels of standardization influenced by regional, cultural, and identity factors.
- **Toxic or Offensive Speech:** The Aya Annotation Platform lacked specific flags for toxic speech, relying on human verification and peer review to mitigate offensive content, but there's no guarantee that all potentially offensive data points were removed during the annotation process.
- **Accounting for mislabeled data:** The Aya Annotation Platform lacks re-labeling capabilities, leading to potential mislabeled data in the Aya Dataset, including instances of incorrect language assignments and non-compliance with instruction-style formatting.


# Additional Information
## Provenance
- **Methods Used:** Crowd-sourced through volunteer annotations, followed by a quality assessment phase in which samples from the dataset were checked.
- **Methodology Details:**
    -  *Source:* Original annotations and edits of opensource NLP datasets
    - *Platform:* [Aya Annotation Platform](https://aya.for.ai/)
    - *Dates of Collection:* May 2023 - Dec 2023


## Dataset Version and Maintenance
- **Maintenance Status:** Actively Maintained
- **Version Details:**
    - *Current version:* 1.0
    - *Last Update:* 02/2024
    - *First Release:* 02/2024
- **Maintenance Plan:** Updates will be periodically made available based on volunteer contributions.


## Authorship
- **Publishing Organization:** [Cohere Labs](https://cohere.com/research)
- **Industry Type:** Not-for-profit - Tech
- **Contact Details:** https://aya.for.ai/


## Licensing Information
This dataset can be used for any purpose, whether academic or commercial, under the terms of the [Apache 2.0](https://opensource.org/license/apache-2-0) License.


## Citation Information
```bibtex
@misc{singh2024aya,
      title={Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning}, 
      author={Shivalika Singh and Freddie Vargus and Daniel Dsouza and Börje F. Karlsson and Abinaya Mahendiran and Wei-Yin Ko and Herumb Shandilya and Jay Patel and Deividas Mataciunas and Laura OMahony and Mike Zhang and Ramith Hettiarachchi and Joseph Wilson and Marina Machado and Luisa Souza Moura and Dominik Krzemiński and Hakimeh Fadaei and Irem Ergün and Ifeoma Okoh and Aisha Alaagib and Oshan Mudannayake and Zaid Alyafeai and Vu Minh Chien and Sebastian Ruder and Surya Guthikonda and Emad A. Alghamdi and Sebastian Gehrmann and Niklas Muennighoff and Max Bartolo and Julia Kreutzer and Ahmet Üstün and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2402.06619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```