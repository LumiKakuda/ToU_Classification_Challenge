Sepsis Survival Prediction using Minimal Clinical Records

This project aims to predict the survival outcome of patients admitted with sepsis potential preconditions using a highly interpretable Logistic Regression model and a minimal feature set (Age, Sex, and Number of Prior Sepsis Episodes).

📊 Dataset

The data used is the Sepsis Survival Minimal Clinical Records dataset.

Source: UCI Machine Learning Repository 

Original Publication: "Survival prediction of patients with sepsis from age, sex, and septic episode number alone" by Chicco and Jurman (2020).

Download Link: https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records

The dataset is highly imbalanced, with over 92% of records belonging to the 'Alive' class. The primary prediction task is binary classification: determining if a patient is alive (1) or deceased (0) approximately 9 days after admission.

💻 Reproduction Guide
To fully reproduce the analysis in this Jupyter Notebook, please follow these steps.

1. Project Structure

├── README.md                                                      # This file
├── Classification Challenge/
  ├── sepsis_classification.ipynb                                  # The Jupyter Notebook containing the analysis
  ├── s41598-020-73558-3_sepsis_survival_study_cohort.csv          #Dataset 1
  └── s41598-020-73558-3_sepsis_survival_validation_cohort.csv     #Dataset 2
  └── s41598-020-73558-3_sepsis_survival_primary_cohort.csv        #Dataset 3
  └── FUNCTIONS_EDA.pv                                             #With Functions used in the jupter notebook
  └── requirements.txt                                             # List of required Python packages

2. Install Dependencies

All necessary Python packages are listed in requirements.txt. 

3. Data Setup

The notebook is configured to read the data directly from the local files.

If the file is already in your directory: No action is needed.

If the file is missing: Download the primary cohort CSV file from the UCI link above or the original zip file and place the primary cohort CSV file in the root directory.

4. Run the Notebook

Start the Jupyter Notebook server:

Execute all cells in the notebook sequentially (using the Cell > Run All option).
