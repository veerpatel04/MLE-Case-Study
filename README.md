# Priceline-MLE-Case-Study

# Hotel Booking Probability Prediction

## Approach and Findings Summary

### Overview
This project develops a machine learning solution to predict hotel booking probabilities using customer search data. The implementation focuses on creating a scalable pipeline that processes raw data, engineers meaningful features, and evaluates multiple models to optimize hotel ranking decisions.

### Exploratory Data Analysis (EDA)
The dataset contains 23 features spanning customer demographics, search parameters, and hotel attributes. Initial analysis revealed a significant class imbalance in the target variable (`bookingLabel`), with non-bookings (label 0) dominating over actual bookings (label 1). As shown in the target distribution plot (Figure 1), approximately 95% of instances represent non-conversions. Numerical features like `numRooms` and `minPrice` showed right-skewed distributions, while `customerReviewScore` displayed a normal distribution pattern (Figure 2).

### Feature Engineering
Key engineered features include temporal metrics such as `days_to_checkin` (calculated from search to check-in dates) and `length_of_stay` (duration between check-in/out dates). A `discount_pct` feature was created to quantify price incentives using the difference between strike and actual prices. Non-predictive identifiers like `searchId` and temporal columns were removed to prevent data leakage.

### Model Development
Three models were implemented in a scikit-learn pipeline:
1. **Logistic Regression** with class weighting
2. **Random Forest** classifier
3. **Gradient Boosting** machine (GBM)

The preprocessing pipeline handles missing values through median/mode imputation, scales numerical features, and one-hot encodes categorical variables like `destinationName` and `deviceCode`. Stratified 3-fold cross-validation was used to account for class imbalance.

### Results
The Random Forest model achieved superior performance with a test AUC of 0.86, demonstrating strong discrimination capability between booking/non-booking events. As shown in the ROC curve comparison (Figure 3), all models significantly outperformed random guessing, with GBM closely following at 0.85 AUC. Feature importance analysis (Figure 4) identified `discount_pct`, `days_to_checkin`, and `customerReviewScore` as top predictors, suggesting price sensitivity and planning horizon significantly influence booking decisions.

---

## Recommendations for Improvement

### 1. Class Imbalance Mitigation
The severe 95:5 class ratio likely limits model sensitivity to booking patterns. Implementing synthetic minority oversampling (SMOTE) or using precision-oriented metrics like F2-score could improve recall of booking events without excessive false positives.

### 2. Model Optimization
Conduct grid searches to optimize hyperparameters like Random Forest's tree depth and GBM's learning rate. Experiment with alternative classifiers like XGBoost or neural networks that might better capture complex feature interactions.

### 3. Enhanced Feature Space
Incorporate external datasets such as:
- Local event calendars to capture demand surges
- Weather forecasts for destination locations
- Historical hotel performance metrics

Create interaction terms between key features like `discount_pct × starLevel` to model premium property price sensitivity.

### 4. Deployment Strategy
Package the model using Docker for containerized deployment on cloud platforms. Implement a FastAPI service with endpoint monitoring to handle real-time prediction requests from the hotel ranking system.

### 5. Performance Monitoring
Establish baselines for feature distributions using tools like Great Expectations. Monitor prediction drift and implement automated retraining triggers when AUC drops below 0.82.

---

## Setup Instructions

### Requirements
- Python 3.8+ environment
- Dependencies:
  ```bash
  pandas==1.4.2
  scikit-learn==1.1.2
  matplotlib==3.5.3
  pyarrow==8.0.0
