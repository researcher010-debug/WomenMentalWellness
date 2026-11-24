# X-WMHEA: Explainable Women's Mental Health and Empowerment Assessment

X-WMHEA is a research implementation for classifying depression-related
outcomes from psychosocial and socio-demographic tabular data. The
framework combines preprocessing, HSIC-LASSO feature selection, a
TabM-style shared-MLP classifier, Adan optimization, and SHAP-based
model interpretation.

> **Research use only.** This software is an experimental research
> implementation. It is not a medical device, does not provide a
> clinical diagnosis, and must not be used as a substitute for
> assessment by a qualified healthcare professional.

## Methodology

The implementation follows the workflow described in the accompanying
research manuscript:

1.  **Preprocessing:** duplicate removal, missing-value imputation,
    target encoding of categorical variables, IQR-based outlier
    handling, and RobustScaler normalization.
2.  **Feature selection:** HSIC-LASSO ranks and selects informative
    psychosocial features.
3.  **Classification:** a TabM-style architecture uses a shared MLP
    backbone with lightweight submodel-specific modulation.
4.  **Optimization:** model parameters are optimized using Adan.
5.  **Explainability:** SHAP can be used to estimate the contribution of
    input features to model predictions.

The two datasets are handled as separate classification experiments:

  -----------------------------------------------------------------------
  Dataset                 Task                    Classes reported in the
                                                  manuscript
  ----------------------- ----------------------- -----------------------
  Women Depression        Binary classification   Not Depressed;
  Dataset                                         Depressed

  Maternal Depression     Four-class              Not Depressed; Severe
  Dataset                 classification          Depressed; Moderate
                                                  Depressed; Mild
                                                  Depressed
  -----------------------------------------------------------------------

## Repository structure

``` text
X-WMHEA/
├── Data/
│   ├── source/
│   │   ├── women_depression.csv
│   │   └── maternal_depression.csv
│   └── preprocessed/
│       ├── women_depression/
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── feature_ranking.csv
│       └── maternal_depression/
│           ├── train.csv
│           ├── test.csv
│           └── feature_ranking.csv
├── model/
│   ├── women_depression/
│   └── maternal_depression/
├── feature_selection.py
├── psychosocial_processing.py
├── architecture.py
├── model_training.py
├── evaluation.py
├── requirements.txt
└── README.md
```

Generated data and model folders are created by the scripts when
required. Do not upload private, restricted, or personally identifiable
records to a public repository or Zenodo record.

## Datasets

The manuscript identifies the following public dataset sources:

-   **Women Depression Dataset:**
    https://www.kaggle.com/datasets/ucimachinelearning/women-depression-dataset
-   **Maternal Depression Dataset:**
    https://data.mendeley.com/datasets/f2r8whrtyx/2

Download the datasets from their original sources and check their
respective terms of use and licenses. Place the CSV files in
`Data/source/` using the filenames expected by the preprocessing script:

-   `Data/source/women_depression.csv`
-   `Data/source/maternal_depression.csv`

The preprocessing configuration currently expects the target column to
be named `target` in both files. If the downloaded CSV uses a different
target-column name, update the corresponding dataset entry in
`psychosocial_processing.py` before running preprocessing. Confirm that
the target labels and feature columns are appropriate for the intended
experiment.

The manuscript reports 6,150 records for the Women Depression Dataset
and 594 records for the Maternal Depression Dataset. Dataset versions
and preprocessing may affect the final usable record counts.

## Environment and installation

-   Python 3.10 or later is recommended.
-   PyTorch can use CUDA when a compatible NVIDIA GPU, driver, and
    CUDA-enabled PyTorch build are available. CPU execution is also
    supported.
-   Install the Python dependencies:

``` bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your
operating system and CUDA version by following the official PyTorch
installation selector: https://pytorch.org/get-started/locally/. The
PyTorch package in `requirements.txt` is a default dependency; replace
it with the matching CUDA-enabled build if needed.

## Running the pipeline

Run the scripts from the repository root in the following order:

### 1. Preprocess both datasets

``` bash
python psychosocial_processing.py
```

This runs preprocessing and HSIC-LASSO feature selection independently
for each dataset. It writes the training and test partitions, feature
rankings, and preprocessing artifacts under `Data/preprocessed/` and
`model/`.

### 2. Train the classifiers

``` bash
python model_training.py
```

The training script trains one model per dataset. It creates a
validation subset from the training partition, selects the best
checkpoint using validation loss, and applies early stopping. The
held-out test partition is not used for checkpoint selection.

Checkpoints and training histories are saved under:

``` text
model/women_depression/
model/maternal_depression/
```

### 3. Evaluate the saved models

``` bash
python evaluation.py
```

Evaluation reports Accuracy, macro Precision, macro Recall, macro
F1-score, and Matthews correlation coefficient (MCC) for the training
and test partitions. It also generates confusion matrices, ROC curves,
precision-recall curves, and training-history plots.

### 4. Generate SHAP explanations

SHAP explanation generation is optional. In `evaluation.py`, set:

``` python
GENERATE_SHAP = True
```

Then run the evaluation script again. Kernel SHAP can be computationally
expensive, particularly for larger datasets.

## Main outputs

For each dataset, the `model/<dataset_name>/` directory may contain:

-   `best_model.pth` --- best model checkpoint selected by validation
    loss.
-   `history.json` and `history.csv` --- training and validation
    history.
-   `training_results.json` --- training-run summary and held-out test
    accuracy/loss.
-   `train_metrics.json` and `test_metrics.json` --- classification
    metrics and class-wise report.
-   `train_confusion_matrix.png` and `test_confusion_matrix.png`.
-   `train_roc_curve.png` and `test_roc_curve.png`.
-   `train_pr_curve.png` and `test_pr_curve.png`.
-   `loss_curve.png` and `accuracy_curve.png`.
-   SHAP summary plots, if SHAP generation is enabled.

Combined training and evaluation summaries are saved in the `model/`
directory.

## Reproducibility and configuration

The implementation defines a random-seed reset function in
`architecture.py`. The training script calls it before each dataset
experiment. Hyperparameters such as batch size, epoch limit, learning
rate, weight decay, validation fraction, and early-stopping patience are
configured near the top of `model_training.py`.

The accompanying manuscript describes the overall methods but does not
specify every implementation-level hyperparameter. Values in the source
code should therefore be treated as implementation settings, not as
manuscript-reported experimental settings. For reproducible publication
results, record the dataset versions, software environment, random seed,
hyperparameters, and any changes to the source code.

## Reported results in the manuscript

The manuscript reports accuracy values of **95.53%** for the Women
Depression Dataset and **96.75%** for the Maternal Depression Dataset.
These are manuscript-reported results, not a guarantee that every
execution of this software will reproduce them. Reproduction depends on
the exact data files, target definition, preprocessing, feature
selection, model configuration, and experimental protocol.

## Data privacy, ethics, and responsible use

The datasets concern sensitive mental-health-related information. Use
them only in accordance with their source licenses, applicable privacy
requirements, and research ethics obligations. Do not attempt to
identify participants. Do not use model predictions to make
consequential decisions about an individual. Performance on these
datasets does not establish clinical validity or generalizability to
other populations.

## Citation

If you use this software, cite the associated research manuscript and
the Zenodo software record. After publishing this repository on Zenodo,
replace the placeholder below with the DOI assigned to the software
record:

``` text
X-WMHEA: Explainable Women's Mental Health and Empowerment Assessment.
Software release, Zenodo. DOI: [insert the assigned Zenodo DOI]
```

Also cite the original datasets at their source pages and follow any
citation requirements specified by their providers.

## License

A license has not been specified in this project description. Before
making the Zenodo record public, add a `LICENSE` file and select a
license that you have the right to apply to the software. The software
license does not override the licenses or terms governing the
third-party datasets.
