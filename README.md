# Costa Rican Household Poverty Prediction

Can supervised machine learning identify vulnerable households from observable socioeconomic characteristics?
This project uses the Inter-American Development Bank's **Costa Rican Household Poverty** dataset (via Kaggle) to classify households by poverty level.

📄 Full write-up: [`reports/Report.pdf`](reports/Report.pdf)

## Key findings

| Task | Best model | Metric |
|---|---|---|
| 4-class (extreme / moderate / vulnerable / non-poor) | Logistic Regression | Macro F1 = 0.42 |
| Binary (poor 1–3 vs non-poor 4) | Logistic Regression | F1 = 0.68, Accuracy = 0.74, AUC = 0.81 |

- The four-class problem performs poorly even with class weighting or SMOTE. A distributional analysis shows that poverty levels 1, 2 and 3 overlap heavily in both feature means and spreads, so this is a limit of the data rather than of the models.
- Most multiclass errors (66%) are between adjacent poverty levels: the models capture the poverty spectrum but cannot draw sharp boundaries within it.
- Reframed as poor vs non-poor, the task becomes learnable. Observable household features are enough to flag households needing assistance, but not to rank the severity of their poverty.

## Repository structure

```
.
├── data/
│   ├── dataset.csv          # Raw data: 9,557 individuals, 2,988 households, 142 features
│   └── features.csv         # Data dictionary (variable name → description)
├── notebooks/
│   └── poverty_prediction.ipynb   # Full pipeline: EDA → cleaning → features → models
├── reports/
│   └── Report.pdf           # Final project report
├── requirements.txt
└── README.md
```

## Methodology

1. **Target fix and household-level split.** Every member gets the household head's target (`parentesco1 == 1`). The 80/20 stratified split is done by household (`idhogar`) to avoid leakage.
2. **Cleaning.** Missing rent, tablet and school-delay values are filled with 0 (absence). `meaneduc` is imputed with the training-set mean. `edjefe`/`edjefa` are merged into a single `education_head` feature, and `dependency` is converted into a numeric ratio.
3. **Feature engineering.** Includes housing vulnerability, asset count, overcrowding and rent per person.
4. **Feature selection.** Average rank across absolute Pearson correlation, mutual information and Random Forest importance, with redundant `SQB*` features removed. The number of features (17) is chosen by cross-validated Logistic Regression.
5. **Modelling.** Logistic Regression, Random Forest, AdaBoost and KNN, tuned with 5-fold stratified `GridSearchCV` on macro F1, under class weighting and SMOTE, for both the multiclass and the binary formulation.

## Getting started

```bash
git clone https://github.com/mireiaacss/costa-rican-household-poverty-prediction.git
cd costa-rican-household-poverty-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/poverty_prediction.ipynb
```

Run the notebook from inside `notebooks/`: it loads the data from `../data/dataset.csv`. A fixed `random_state=42` is used throughout for reproducibility.

## Data source

Inter-American Development Bank, *Costa Rican Household Poverty Level Prediction*, Kaggle.
The data is used here for educational purposes. See the Kaggle competition page for its terms of use.
