## Tsunami Risk Classification – Machine Learning Models & Streamlit App

### Problem Statement
The goal of this project is to predict whether an earthquake event is likely to trigger a tsunami using multiple machine learning models. The dataset contains seismic features such as magnitude, depth, latitude/longitude, and earthquake impact indicators collected from January 1, 2001 to December 31, 2022 (22 years)

This project demonstrates the full end‑to‑end ML workflow — model training, evaluation, deployment, and interactive visualization using Streamlit.

---------------------------------------------------------------------------------------------------------------------------------------

### Dataset Description
The dataset used is the Global Earthquake–Tsunami Risk Assessment Dataset (from Kaggle).
It contains 782 instances and 13 numerical features (Detailed Feature description ahead).

Source : https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset 

- File Format: CSV
- File Size: ~41KB
- Target Variable: Tsunami potential indicator (binary classification) with values (0/1)
- Balanced Dataset: Suitable for binary classification tasks

##### Geographical specifications of data collected
* Seismic Magnitude Distribution
* Range: 6.5 - 9.1 Richter scale
* Mean Magnitude: 6.94
* Data from January 1, 2001 to December 31, 2022 (22 years)
* Major Earthquakes (≥8.0): 28 events including the 2004 (9.1) and 2011 (9.1) mega-earthquakes

---------------------------------------------------------------------------------------------------------------------------------------

### Feature Description

| Feature   | Type    | Description                                       | Range / Values                 | Tsunami Relevance |
|-----------|---------|---------------------------------------------------|--------------------------------|--------------------|
| magnitude | Float   | Earthquake magnitude (Richter scale)              | 6.5 – 9.1                      | High – Primary tsunami predictor |
| cdi       | Integer | Community Decimal Intensity (felt intensity)      | 0 – 9                          | Medium – Population impact measure |
| mmi       | Integer | Modified Mercalli Intensity (instrumental)        | 1 – 9                          | Medium – Structural damage indicator |
| sig       | Integer | Event significance score                          | 650 – 2910                     | High – Overall hazard assessment |
| nst       | Integer | Number of seismic monitoring stations             | 0 – 934                        | Low – Data quality indicator |
| dmin      | Float   | Distance to nearest seismic station (degrees)     | 0.0 – 17.7                     | Low – Location precision |
| gap       | Float   | Azimuthal gap between stations (degrees)          | 0.0 – 239.0                    | Low – Location reliability |
| depth     | Float   | Earthquake focal depth (km)                       | 2.7 – 670.8                    | High – Shallow = higher tsunami risk |
| latitude  | Float   | Epicenter latitude (WGS84)                        | -61.85° to 71.63°              | High – Ocean proximity indicator |
| longitude | Float   | Epicenter longitude (WGS84)                       | -179.97° to 179.66°            | High – Ocean proximity indicator |
| Year      | Integer | Year of occurrence                                | 2001 – 2022                    | Medium – Temporal patterns |
| Month     | Integer | Month of occurrence                               | 1 – 12                         | Low – Seasonal analysis |
| tsunami   | Binary  | Tsunami potential (**TARGET**)                    | 0, 1                           | TARGET VARIABLE |


---------------------------------------------------------------------------------------------------------------------------------------

### Machine Learning Models Used
Six classification models were trained on the same dataset:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Random Forest (Ensemble)
* XGBoost (Ensemble)

Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

##### All trained models are saved inside the model/ folder for reuse in the Streamlit app.

---------------------------------------------------------------------------------------------------------------------------------------

### Model Performance Comparison

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|----------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression  | 0.8481 |	0.9368  |	0.77142 |	0.8709  |	0.81818 |	0.6923 |
| Decision Tree        | 0.9113   |	0.8985  |	0.9285  |	0.8387  |	0.8813  |	0.8136  |
| K-Nearest Neighbors  | 0.9367 |	0.9455  |	0.9642  |	0.8709  |	0.9152  |	0.8678    |
| Naive Bayes (Gaussian) | 0.8101   |	0.83938 |	0.7222  |	0.8387  |	0.7761  |	0.6180   |
| Random Forest (Ensemble) | 0.9747    |	0.9852  |	0.96774 |	0.9677  |	0.9677  |	0.9469  |
| XGBoost (Ensemble)   | 0.9873 |	0.98454 |	0.9688  |	1   |	0.9841  |	0.9739  |

---------------------------------------------------------------------------------------------------------------------------------------

### Observations on Model Performance

| ML Model Name        | Observation |
|----------------------|-------------|
| Logistic Regression  | Shows steady, reliable performance with good AUC and F1 scores, but it struggles to capture complex non‑linear relationships in the data. Performs well but is clearly outperformed by tree‑based and ensemble models. |
| Decision Tree        | Performs well with strong accuracy and MCC, but slightly lower AUC than the ensembles indicates some overfitting. It captures non‑linear patterns better than logistic regression but is less stable than Random Forest and XGBoost. |
| K-Nearest Neighbors  | Good balance between precision and recall with high AUC. kNN does well after scaling, but performance is limited in comparison with tree ensembles because it is sensitive to local variations and distance metrics. |
| Naive Bayes (Gaussian) | Lowest performance among all models, mainly because its assumptions (feature independence + Gaussian distribution) don't hold well for this dataset. Still gives a decent baseline but misses complex feature interactions. |
| Random Forest (Ensemble) | Strong overall performance with very high scores across all metrics. It generalizes well and handles non‑linear patterns effectively. Nearly top‑tier, only slightly behind XGBoost. |
| XGBoost (Ensemble)   | Best performing model across almost all metrics. Achieves highest accuracy, F1, and MCC, and perfect recall, indicating it can capture all tsunami‑positive cases. Shows excellent generalization with minimal overfitting. |

---------------------------------------------------------------------------------------------------------------------------------------

### Streamlit App Features
The deployed app allows you to:

- Upload test CSV data
- Select one of the six pre‑trained models
- View evaluation metrics (if your CSV includes ground-truth tsunami)
- View confusion matrix & classification report
- Download prediction outputs


---------------------------------------------------------------------------------------------------------------------------------------

### Project Structure
project-folder/

│-- app.py

│-- train_models.py

│-- requirements.txt

│-- README.md

│-- model/

│ ----------  ├── logistic_regression.pkl

│ ----------  ├── decision_tree.pkl

│  ---------- ├── knn.pkl

│ ----------  ├── gaussian_nb.pkl

│ ----------  ├── random_forest.pkl

│ ----------  ├── xgboost.pkl

│ ----------  ├── feature_list.json

│ ----------  └── metrics.csv

---------------------------------------------------------------------------------------------------------------------------------------

### How to Run Locally

Install dependencies:
pip install -r requirements.txt

And, Run the Streamlit app:
streamlit run app.py

Finally, Upload your test CSV and explore the results.

---------------------------------------------------------------------------------------------------------------------------------------

### Live App Link
(Insert your Streamlit deployment link here)


---------------------------------------------------------------------------------------------------------------------------------------

### GitHub Repository
(Insert your GitHub repo link here)