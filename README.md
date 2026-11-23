# PropValue AI: Machine Learning-Based Real Estate Valuation

Experimental educational project that predicts house price per unit area from property
characteristics using regression models. The workspace and data were prepared on
Microsoft Azure Machine Learning (Azure for Students).

## Dataset
UCI Real Estate Valuation dataset (414 records, Sindian District, New Taipei City, Taiwan), CC BY 4.0.
https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

## How to run
pip install -r requirements.txt
python train_local.py --data data/real_estate_valuation.csv

Outputs (metrics CSVs and plots) are saved to the results/ folder.

## Method
80/20 train-test split, 5-fold cross-validation, five regression models
(Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting), metrics RMSE, MAE and R2,
and permutation feature importance.

Educational model only, not a replacement for professional property valuation.
