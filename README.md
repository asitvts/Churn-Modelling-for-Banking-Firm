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

## Observations and Insights

### 1. **Geographical Influence on Churn**
One of the most striking findings is that customer churn varies significantly by geography. 

- **Germany**:
  - **1695** people stayed, **814** people left.
  - **32.44%** of German customers left the bank.

- **France**:
  - **4204** people stayed, **810** people left.
  - **16.15%** of French customers left the bank.

- **Spain**:
  - **2064** people stayed, **413** people left.
  - **16.67%** of Spanish customers left the bank.

#### 🔍 **Key Observation:**
- The churn rate for **German customers (32.44%) is nearly double** that of **French (16.15%) and Spanish (16.67%)** customers.
- The bank is struggling to retain German customers compared to its Spanish and French counterparts.
- This suggests a **potential underlying issue** with customer satisfaction or services provided to German clients.

---

### 2. **Age Factor in Customer Churn**
Age also plays a significant role in customer retention.

- The **youngest customer** is **18 years old**, and the **oldest** is **92 years old**.
- **Average age of customers who left**: **44.84 years**
- **Average age of customers who stayed**: **37.41 years**

#### 🔍 **Key Observation:**
- The bank is **successfully retaining younger clients**.
- However, **older customers (mid-40s and above) are more likely to leave**.
- This indicates a potential gap in services tailored to older customers.

---

### 3. **Bank Balance and Churn Relationship**
- **Average balance of customers who left**: **91,108.54**
- **Average balance of customers who stayed**: **72,745.30**

#### 🔍 **Key Observation:**
- **Customers with higher bank balances are leaving the bank more frequently**.
- This is **bad news for business**, as high-balance customers have greater financial activity, which could drive profits.
- Retaining such clients should be a priority for the bank.

---

### 4. **Credit Score Analysis**
- **Average credit score of customers who left**: **645.35**
- **Average credit score of customers who stayed**: **651.85**

#### 🔍 **Key Observation:**
- The difference in credit scores is **not very significant**.
- However, **people with lower credit scores are slightly more likely to leave**.
- While credit score **isn’t a major deciding factor**, it still contributes to churn prediction.

---

### **Conclusion**
- The bank needs to **improve customer retention strategies for German clients**.
- **Older customers (mid-40s and above) are at higher risk of leaving**—targeted engagement programs can help.
- **High-balance customers are churning more**, which directly affects profitability—premium services or loyalty programs could help retain them.
- While **credit score isn’t a strong churn predictor**, it still shows a slight trend towards lower-scoring customers leaving.

The insights from this analysis can help the bank develop strategies to improve customer satisfaction and reduce churn effectively. 🚀

This project demonstrates the effectiveness of different machine learning models in predicting customer churn. 

- **Logistic Regression** was simple but had lower accuracy.
- **KNN** performed decently but was slow for large data.
- **Decision Tree** was effective but overfitted without tuning.
- **Random Forest** improved performance using multiple trees.
- **GridSearchCV Decision Tree** gave the best results after hyperparameter tuning.

Further improvements can be made using **ensemble methods** (boosting, bagging) or **deep learning models**.

---

## 📝 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
### 💡 Interested in this project? Feel free to contribute and expand upon these findings!

