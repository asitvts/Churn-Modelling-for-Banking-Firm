# Bank Customer Churn Prediction

## Project Overview
This project aims to predict whether a bank's customer will churn (leave) or stay based on various features extracted from a given dataset. The dataset was initially in CSV format and contained the following information:

- **Row Number**: Unique index of the record
- **Customer ID**: Unique identifier for each customer
- **Surname**: Last name of the customer
- **Geographical Location**: Country of the customer (e.g., France, Germany, Spain)
- **Gender**: Male or Female
- **Credit Score**: A numerical score indicating creditworthiness
- **Number of Products**: Number of financial products used by the customer
- **Is Active Member**: Whether the customer is actively engaging with the bank
- **Account Balance**: The total balance in the customer's account
- **Tenure**: Number of years the customer has been with the bank
- **Estimated Salary**: Approximate annual salary of the customer

The goal is to build machine learning models to predict customer churn based on these features.

## Models Implemented
The following models were developed and evaluated:

1. **Logistic Regression** (`churn-bank-customers-logistic-regression.ipynb`)
2. **K-Nearest Neighbors (KNN)** (`churn-bank-customers-knn.ipynb`)
3. **Decision Tree Classifier** (`churn-bank-decision-tree.ipynb`)
4. **Random Forest Classifier** (`bank-churn-random-forest.ipynb`)
5. **Decision Tree with Grid Search CV** (`bank-churn-decision-tree-gridsearchcv.ipynb`)

Each model was trained, tested, and evaluated using various performance metrics like accuracy, precision, recall, and F1-score.

---

## Observations from Decision Tree Grid Search CV Model

### **1. Hyperparameter Tuning Results**
The Decision Tree model was optimized using **Grid Search Cross-Validation (GridSearchCV)** to find the best hyperparameters:

- **Best Criterion**: The model performed better with **entropy** as the splitting criterion rather than gini.
- **Optimal Max Depth**: A **max depth of 6-8** was found to be the best, preventing overfitting.
- **Min Samples Split**: Setting a minimum sample split of **10-20** helped improve generalization.
- **Best Parameters Found:**
  ```python
  {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 15}
  ```

### **2. Model Performance**
After tuning, the Decision Tree model showed the following performance:

- **Accuracy**: Improved from ~78% to **82%**
- **Precision**: Increased slightly, meaning fewer false positives
- **Recall**: Higher recall indicates better customer churn detection
- **F1-Score**: Achieved a more balanced performance

### **3. Feature Importance**
The most important features influencing churn prediction were:

1. **Number of Products**: Customers with **only one product** had the highest churn rate.
2. **Is Active Member**: Non-active members were far more likely to churn.
3. **Credit Score**: While important, its impact was less compared to other factors.
4. **Geographical Location**: Customers from Germany had a significantly higher churn rate than those from France or Spain.

### **4. Overfitting & Generalization**
- Before hyperparameter tuning, the **training accuracy was 95%** while test accuracy was only **78%**, indicating overfitting.
- After tuning with **max depth & min samples split**, the test accuracy increased to **82%**, meaning better generalization.

---

## How to Run the Project

1. Clone this repository:
   ```sh
   git clone <repo-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd bank-churn-prediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Open and run the Jupyter notebooks in the following order for best results:
   - `churn-bank-customers-logistic-regression.ipynb`
   - `churn-bank-customers-knn.ipynb`
   - `churn-bank-decision-tree.ipynb`
   - `bank-churn-random-forest.ipynb`
   - `bank-churn-decision-tree-gridsearchcv.ipynb`

---

## Conclusion
This project demonstrates the effectiveness of different machine learning models in predicting customer churn. 

- **Logistic Regression** was simple but had lower accuracy.
- **KNN** performed decently but was slow for large data.
- **Decision Tree** was effective but overfitted without tuning.
- **Random Forest** improved performance using multiple trees.
- **GridSearchCV Decision Tree** gave the best results after hyperparameter tuning.

Further improvements can be made using **ensemble methods** (boosting, bagging) or **deep learning models**.
