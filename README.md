# PetFinder Multimodal Adoption Speed Prediction

This project predicts the `AdoptionSpeed` category (0–4) using pet tabular attributes, images, and description text from the [PetFinder Adoption Prediction Kaggle competition](https://www.kaggle.com/competitions/petfinder-adoption-prediction).

## View results without running anything

Open [`PetFinder_Multimodal_Prediction_GitHub.ipynb`](PetFinder_Multimodal_Prediction_GitHub.ipynb) on GitHub. The notebook contains **saved outputs**, including final evaluation tables and four figures. You do not need Google Drive access to view these results.

## Methods

- Tabular: numeric imputation/scaling, categorical imputation/one-hot encoding, Random Forest.
- Image: pretrained frozen ResNet18 512-dimensional embeddings and Logistic Regression.
- Text: pretrained frozen `bert-base-multilingual-cased` 768-dimensional mean-pooled embeddings and Logistic Regression.
- Early fusion: concatenate the three feature representations and fit regularized Logistic Regression.
- Late fusion: equal-weight averaging of probabilities from the three single-modality classifiers.

The working dataset contains 14,640 records, split into training (10,248), validation (2,196) and test (2,196). The validation set was used for tuning early-fusion regularization; the test set was used for final evaluation.

## Saved test results

| Model | Accuracy | Macro F1 | Quadratic Weighted Kappa |
|---|---:|---:|---:|
| Tabular-only | 0.3884 | 0.3325 | 0.3246 |
| Image-only | 0.2910 | 0.2587 | 0.2419 |
| Text-only | 0.3024 | 0.2714 | 0.1410 |
| Early fusion | 0.3689 | 0.3222 | 0.3199 |
| Late fusion | 0.3552 | 0.3130 | 0.2613 |

These values are recorded from the original notebook execution. The GitHub-ready notebook was edited for presentation and has **not** been rerun end-to-end after removing Google Drive backup steps.

## Reproducing the experiment

1. Open the notebook in **Google Colab**, preferably with a GPU runtime. Install the dependencies in `requirements.txt` if missing; Colab versions may differ.
2. Join/accept the Kaggle competition's data-access rules and obtain a Kaggle API token. Store it in **Colab Secrets** under `KAGGLE_API_TOKEN`; never paste a token directly into the notebook or commit credentials.
3. Run cells in order. They download competition files into `/content/petfinder`, extract training images, build splits, extract ResNet18 and multilingual BERT embeddings, fit the models, and generate the results and figures.
4. Expect substantial image download, disk use, and GPU inference time. The notebook saves generated artifacts locally under `/content/petfinder/results` and `/content/petfinder/figures`. These paths are temporary in Colab; download any files you wish to keep.

**Limitations:** Kaggle access and a working Colab environment are required to rerun the notebook; a fresh run is not guaranteed across changing package versions. The current notebook retains exploratory download steps. PetID-to-embedding alignment and the complete fresh-runtime reproducibility path have not been independently audited. Do not interpret the saved outputs as proof of an end-to-end rerun.

## Privacy and sharing

Do not upload Kaggle API credentials, personal Drive data, raw competition archives, or large pretrained/model artifacts to this public repository. Follow Kaggle's dataset licensing and competition rules when sharing data.
