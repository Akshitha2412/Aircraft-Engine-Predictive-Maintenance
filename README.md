# Aircraft-Engine-Predictive-Maintenance
This project is an Predictive maintenance system for aircraft engines using Machine Learning and KMeans Clustering. The system analyzes aircraft engine sensor data to predict engine health conditions, Remaining Useful Life (RUL), and maintenance alerts.


### 1. Data Collection
Collected aircraft engine operational and sensor datasets containing engine cycles, settings, and sensor readings.

### 2. Data Understanding and Exploration
Analyzed dataset structure, features, sensor behavior, and engine operational cycles.

### 3. Data Preprocessing
Prepared the dataset using StandardScaler and PowerTransformer for numerical feature transformation.

### 4. Feature Engineering
Created additional features such as RUL, Health Index, Health State, and Alert.

### 5. Remaining Useful Life (RUL) Calculation
Calculated RUL by subtracting the current cycle from the maximum engine cycle.

### 6. Feature Scaling using StandardScaler
Normalized sensor values to bring all features into a common scale.

### 7. Data Transformation using PowerTransformer
Reduced skewness and improved feature distribution for better model performance.

### 8. Building Preprocessing Pipeline and ColumnTransformer
Automated preprocessing workflow and applied transformations to numerical columns.

### 9. Train-Test Split
Divided the dataset into training and testing sets for model evaluation.

### 10. Machine Learning Model Building
Implemented multiple regression models for predictive maintenance analysis.

### 11. Linear Regression Implementation
Built a baseline regression model for RUL prediction.

### 12. Random Forest Regressor Implementation
Used ensemble learning with multiple decision trees for improved prediction accuracy.

### 13. Support Vector Regressor (SVR) Implementation
Applied SVR for capturing complex nonlinear relationships in the dataset.

### 14. XGBoost Regressor Implementation
Implemented  boosting for high-performance RUL prediction.

### 15. Model Evaluation using R² Score, MSE, and MAE
Evaluated prediction accuracy and model performance using regression metrics.

### 16. Cross Validation
Applied K-Fold Cross Validation to improve model reliability and stability.

### 17. Hyperparameter Tuning using RandomizedSearchCV
Optimized model parameters to achieve better performance and reduce overfitting.

### 18. Final Model Selection (XGBoost)
Selected XGBoost as the best-performing model based on evaluation results.

### 19. PCA for Dimensionality Reduction
Reduced high-dimensional sensor data into fewer components for clustering and visualization.

### 20. KMeans Clustering for Health State Classification
Grouped engines into Healthy, Warning, and Critical health states based on sensor behavior.

### 21. Health Index Calculation
Calculated engine health score based on engine degradation progression.

### 22. Maintenance Alert Generation
Generated maintenance recommendations such as Normal, Inspection Required, and Immediate Maintenance.

### 23. Streamlit Dashboard Development
Developed an interactive dashboard for prediction, monitoring, and visualization.

### 24. Real-Time Prediction and Visualization
Enabled users to input sensor values and visualize engine health conditions in real time.

### 25. Model Deployment on Hugging Face / Streamlit Cloud
Deployed the predictive maintenance system for online access and real-time usage.
