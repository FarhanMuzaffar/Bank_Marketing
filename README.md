# **Bank Marketing Campaign: Predicting Term Deposit Subscriptions**

## **Project Title and Description**

This project aims to develop a predictive model to determine whether a client will subscribe to a term deposit, based on data from a Portuguese bank's telemarketing campaign. The analysis involves exploring the dataset, engineering relevant features, selecting and training machine learning models, and evaluating their performance. The insights and final model can assist in effectively targeting potential customers in future marketing campaigns.

## **Dataset Description**

### **Dataset Overview**
The dataset used in this project is the Bank Marketing dataset, as detailed by Moro et al. (2014). It includes data from a Portuguese bank’s telemarketing campaign, enriched with social and economic context attributes. The binary classification goal is to predict whether a client will subscribe to a term deposit (output variable `y`).

### **Sources**
The dataset was created by Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho), and Paulo Rita (ISCTE-IUL) in 2014.

### **Number of Instances and Attributes**
- **Instances**: 41,188
- **Attributes**: 21 (20 input variables and 1 output variable)

### **Input Variables**
1. **age**: Numeric.
2. **job**: Type of job (categorical).
3. **marital**: Marital status (categorical).
4. **education**: Education level (categorical).
5. **default**: Has credit in default? (categorical).
6. **housing**: Has housing loan? (categorical).
7. **loan**: Has personal loan? (categorical).
8. **contact**: Contact communication type (categorical).
9. **month**: Last contact month of the year (categorical).
10. **day_of_week**: Last contact day of the week (categorical).
11. **duration**: Last contact duration, in seconds (numeric).
12. **campaign**: Number of contacts performed during this campaign (numeric).
13. **pdays**: Number of days since last contacted from a previous campaign (numeric).
14. **previous**: Number of contacts performed before this campaign (numeric).
15. **poutcome**: Outcome of the previous marketing campaign (categorical).
16. **emp.var.rate**: Employment variation rate - quarterly indicator (numeric).
17. **cons.price.idx**: Consumer price index - monthly indicator (numeric).
18. **cons.conf.idx**: Consumer confidence index - monthly indicator (numeric).
19. **euribor3m**: Euribor 3 month rate - daily indicator (numeric).
20. **nr.employed**: Number of employees - quarterly indicator (numeric).

### **Output Variable**
21. **y**: Has the client subscribed to a term deposit? (binary: "yes", "no").

### **Missing Values**
Some categorical attributes contain missing values coded as "unknown". These missing values can be handled using deletion or imputation techniques.

## **Project Structure**

The project is organized into four main Jupyter notebooks, each representing a critical phase in the machine learning pipeline:

1. **Data Collection and EDA**: This notebook covers data loading, initial exploration, and detailed exploratory data analysis (EDA) to uncover relationships and trends within the dataset.
2. **Feature Engineering**: This notebook details the creation of new features, including interaction terms and binning of continuous variables, followed by feature selection using L1 regularization.
3. **Model Selection and Training**: This notebook involves training various machine learning models, including Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM). Hyperparameter tuning is performed using GridSearch with Stratified K-Fold.
4. **Model Evaluation**: This notebook evaluates the performance of the trained models using metrics such as ROC AUC score and Average Precision (AP). It also compares the final models against a baseline model.

## **Installation Instructions**

To run this project, ensure that you have Python 3.x installed on your system. Install the required dependencies using the following steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/FarhanMuzaffar/bank-marketing.git
   ```
2. Navigate to the project directory:
   ```bash
   cd bank-marketing-campaign
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## **Usage Instructions**

To run the notebooks, follow these steps:

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run the notebooks in the following order:
   - `1_Data_Collection_and_EDA.ipynb`
   - `2_Feature_Engineering.ipynb`
   - `3_Model_Selection_and_Training.ipynb`
   - `4_Model_Evaluation.ipynb`

Follow the instructions within each notebook to reproduce the results.

## **Results and Findings**

### **Initial Data Insights**
- **Feature Importance**: Categorical variables like `job`, `marital`, `education`, `contact`, `month`, and `poutcome` show significant variations in relation to the target variable `y`. Continuous variables like `duration`, `campaign`, `previous`, `emp.var.rate`, `cons.conf.idx`, `euribor3m`, and `nr.employed` also have notable relationships with the target.
- **Correlation Among Categorical Features**: The custom function `categorical_levels_corr()` revealed significant relationships between categorical feature levels, with a threshold of 3.99 chosen to determine significance.

### **Feature Engineering Insights**
- **Interaction Terms**: New interaction terms were created for highly correlated categorical feature pairs, enhancing the model's ability to capture complex relationships.
- **Binning Continuous Features**: Continuous features were binned into categories, reducing outlier impact and capturing non-linear relationships.
- **L1 Regularization**: Applied to select important features, reducing the feature count from 228 to a more manageable number.
- **Duration Feature Removal**: The `duration` feature was removed as it provided unrealistic predictive power, ensuring a more realistic model.

### **Model Performance**
- **Logistic Regression**: Achieved a ROC AUC score of 0.823.
- **Random Forest**: Slightly outperformed Logistic Regression with a ROC AUC score of 0.825.
- **Decision Tree & KNN**: Performed poorly due to overfitting and handling imbalanced data.
- **SVM**: Not feasible due to computational limitations.

### **Evaluation Metrics**
- **ROC AUC Score**: Chosen as the primary metric due to the imbalanced nature of the dataset. The final models achieved a ROC AUC score of 0.83, a significant improvement from the baseline model's 0.50.
- **Average Precision (AP)**: Improved from a chance level AP of 0.11 to 0.62, indicating better handling of precision across recall levels.

### **Conclusion**
The project successfully identified Logistic Regression and Random Forest as the most effective models for predicting term deposit subscriptions, offering a significant improvement over the baseline model. The findings underscore the importance of feature engineering and hyperparameter tuning in building robust predictive models.

## **Future work and improvements.**