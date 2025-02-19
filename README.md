# Priceline-MLE-Case-Study

# Hotel Booking Probability Prediction

## Approach and Findings Summary

### Overview
This project develops a machine learning solution to predict hotel booking probabilities using customer search data. The implementation focuses on creating a scalable pipeline that processes raw data, engineers meaningful features, and evaluates multiple models to optimize hotel ranking decisions.

### Exploratory Data Analysis (EDA)
The dataset contains 23 features spanning customer demographics, search parameters, and hotel attributes. Initial analysis revealed a significant class imbalance in the target variable (`bookingLabel`), with non-bookings (label 0) dominating over actual bookings (label 1). As shown in the target distribution plot, approximately 95% of instances represent non-conversions. Numerical features like `numRooms` and `minPrice` showed right-skewed distributions, while `customerReviewScore` displayed a normal distribution pattern.

### Feature Engineering
Key engineered features include temporal metrics such as `days_to_checkin` (calculated from search to check-in dates) and `length_of_stay` (duration between check-in/out dates). A `discount_pct` feature was created to quantify price incentives using the difference between strike and actual prices. Non-predictive identifiers like `searchId` and temporal columns were removed to prevent data leakage.

### Model Development and Results
We evaluated three machine learning models for predicting booking probability:

Logistic Regression achieved an AUC of 0.58
Random Forest achieved an AUC of 0.60
Gradient Boosting achieved an AUC of 0.65

Gradient Boosting demonstrated the strongest performance, though there remains room for improvement. The analysis identified key predictive features, with review count, customer review scores, and star level showing the strongest influence on booking probability. Feature importance analysis identified `discount_pct`, `days_to_checkin`, and `customerReviewScore` as top predictors, suggesting price sensitivity and planning horizon significantly influence booking decisions.

---

## Recommendations for Improvement
### Data and Features

Track major holidays, local events, and peak travel seasons to capture temporal patterns that influence booking behavior.
Incorporate historical booking rates and patterns to provide context for predicting future bookings.
Add competitor price tracking to understand each hotel's relative value proposition within its market segment.
Integrate local event data and geographic features to provide better context for booking predictions.

### Model Enhancements

Implement SMOTE or adjusted class weights to address the significant imbalance between booked and non-booked hotels.
Upgrade to more sophisticated algorithms like XGBoost or LightGBM to better capture complex booking patterns.
Combine predictions from multiple models through stacking or ensemble methods to improve overall accuracy.
Set up automated hyperparameter optimization to maintain peak model performance as data patterns change.

### System Improvements

Optimize the feature engineering pipeline to enable real-time predictions for incoming search requests.
Implement continuous model monitoring to track performance metrics and trigger retraining when needed.
Create an A/B testing framework to safely evaluate new features and model improvements.
Develop API endpoints for both real-time predictions and batch processing capabilities.

### Future Work

Build a price sensitivity model to optimize pricing strategies in conjunction with booking predictions.
Create a personalization system that incorporates individual user preferences and booking history.
Implement an automated feedback loop to incorporate booking outcomes into regular model updates.
Expand the system to handle multiple markets with region-specific features and models.

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
