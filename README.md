# Do LLM hallucination detectors suffer from low-resource effect?

Long Paper accepted at the **EACL 2026 Main Conference!** <a href="https://aclanthology.org/2026.eacl-long.136/">[Paper]</a> <a href="https://drive.google.com/file/d/1U7dOsSGxOzR9QbdeeDkWbd7HjW6lWH5f/view?usp=drive_link">[Slides]</a>. This repository contains the dataset along with all scripts developed in this work for data curation, prompting, hallucination detection, cross-lingual and multilingual experiments, and entropy analyses.

## Dataset: mTREx

**mTREx** is a multilingual extension of the original TREx factual QA benchmark. It has been introduced in this work and created via careful translation and human validation.

* **Languages**: English (EN), German (DE), Hindi (HI), Bengali (BN), Urdu (UR)

Each language directory in `data/mTREx/` contains the QA pairs for that particular language.

## Prompts (`prompts/`)

| File                 | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `GMMLU_prompts.json` | Prompt templates used for Global-MMLU multiple-choice evaluation |
| `mTREx_prompts.json` | Prompt templates used for factual QA on mTREx        |

## Scripts and Codebase

### Data Curation Scripts (`scripts/data_curation_scripts/`)

| Script           | Description                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `trex_parser.py` | Parses the original TREx dataset using Python's `multiprocessing` to extract specific factual relations (*Capitals*, *Country*, and *Official Language*) |
| `translator.py`  | Translates English TREx questions into target languages using *GPT-4o-mini* from OpenAI API |

### Sampling-Based Black-Box Hallucination Detection Methods (`scripts/Sampling_HD_methods/`)

These methods rely **only on model-generated responses**, without access to internal artifacts / parameters.

| Script               | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `selfcheckGPT.py`    | Implementation of SelfCheckGPT method (SGM) [<a href="https://doi.org/10.18653/v1/2023.emnlp-main.557">[Paper]</a>] |
| `semanticEntropy.py` | Implementation of Semantic Entropy–based method (SEM) [<a href="https://doi.org/10.1038/s41586-024-07421-0">[Paper]</a>] |

### Model-Artifact-Based Hallucination Detection Methods (MAM) (`scripts/MAM_HD_methods/`)

These methods leverage **internal model artifacts** collected during generation.

| Script                                             | Description                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `ModelArtifacts.py`                                | Extracts self-attention scores and fully connected activations from LLMs |
| `ModelArtifacts_Classifier.py`                     | Trains and evaluates classifiers based on model artifacts in same-language settings (MAM detectors) |
| `ModelArtifacts_Classifier_Cross_lingual.py`       | Cross-lingual (EN → target language) training and evaluation for MAM detectors |
| `ModelArtifacts_Classifier_Multi_lingual.py`       | Multilingual training training and evaluation for MAM detectors  |
| `ModelArtifacts_Classifier_entropy_cal.py`         | Computes entropy of softmax distributions for detectors and generators (LLMs) |
| `ModelArtifacts_multiple_gen_tokens.py`            | Extracts artifacts averaged across multiple generated tokens (MAM multi-token variant) |
| `ModelArtifacts_Classifier_multiple_gen_tokens.py` | Classifiers for MAM multi-token variant |

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aisoc-lab/low-resource-hallucination-detection.git
cd low-resource-hallucination-detection
```

### 2. Create a Conda Environment (Recommended)

```bash
conda create --name hallu python==3.12
conda activate hallu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## License

This project is released under the license specified in the `LICENSE` file.

## Citation
If you find this research useful in your work or you use this dataset / code, please cite the following paper:
```
@inproceedings{datta-etal-2026-llm,
    title = "Do {LLM} hallucination detectors suffer from low-resource effect?",
    author = "Datta, Debtanu  and
      Chilukuri, Mohan Kishore  and
      Kumar, Yash  and
      Ghosh, Saptarshi  and
      Zafar, Muhammad Bilal",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.136/",
    doi = "10.18653/v1/2026.eacl-long.136",
    pages = "2959--2985",
    ISBN = "979-8-89176-380-7"
}
```
