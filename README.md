# Data-driven Hybrid Machine Learning and Neural Network Approach for Predicting Rainfall Patterns with Feature Engineering


## Environment Requirement
- Python==3.12.0
- torch==2.5.1
- numpy==2.1.2
- pandas==2.2.3
- scikit-learn==1.5.2
- seaborn==0.13.2
- matplotlib==3.9.2
- missingno==0.5.2

## Dataset
`dataset/weatherAUS.csv`: public rain condition dataset containing 10 years of daily weather observations measured in numerous weather stations.

## Code Organization
The script to reproduce all the figures, and tables in the paper is as follows:
- `feature_engineering.ipynb`: data processing and feature engineering
- `PCA.ipynb`: PCA analysis on top of the previous feature selection/reconstruction in feature engineering
- `complete_process.ipynb`: basic overall demo for this work
- `roc_analysis_weak_ml_classifier.py`: training setting search by ROC analysis for multiple classifiers below:
     - Decision Tree (DT)  
     - Random Forest (RF)  
     - Logistic Regression (LR)  
     - Naive Bayes (NB)  
     - K-Nearest Neighbors (KNN)  
     - Gradient Boosting (GB)  
     - Voting Classifiers (DT+LR+RF and KNN+LR+RF)
- `roc_analysis_lasso.py`: training setting search and ROC analysis for the LASSO model.
- `roc_analysis_mlp.py`: training setting search and ROC analysis for the MLP model.
- `roc_analysis_lstm.py`: training setting search and ROC analysis for the LSTM model.
- `roc_analysis_kan.py`: training setting search and ROC analysis for the Kolmogorov-Arnold Network (KAN).
