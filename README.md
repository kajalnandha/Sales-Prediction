Here’s a rephrased and structured version of your README for GitHub:

---

# ML Exercise: Advertising Sales Prediction with Simple Linear Regression  

### Dataset  
The dataset used for this project is from Kaggle: [Advertising Dataset](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input). It includes advertising budgets across TV, Radio, and Newspaper mediums, along with the corresponding sales figures.

---

## Project Overview  
This project focuses on analyzing the Advertising Dataset and building a simple linear regression model to predict sales based on advertising spend. The primary goal is to explore the relationship between TV advertising budgets and sales and evaluate the predictive capability of a linear regression model.  

---

## Author  
**Name:** Kajal Nandha  
**Email:** kaajunanda@gmail.com  

---

## Objectives  
The objectives of this project are:  
1. To analyze the relationship between advertising spend on TV and sales.  
2. To build and train a simple linear regression model to predict sales based on TV advertising budgets.  
3. To evaluate the model's performance using key statistical metrics.  
4. To provide detailed statistical insights into the regression model.  

---

## Dataset Description  
The dataset contains the following columns:  
- **TV:** Advertising budget for TV (in thousands of dollars).  
- **Radio:** Advertising budget for Radio (in thousands of dollars).  
- **Newspaper:** Advertising budget for Newspaper (in thousands of dollars).  
- **Sales:** Sales generated (in thousands of units).  

**Sample Data:**  

| TV    | Radio | Newspaper | Sales |  
|-------|-------|-----------|-------|  
| 230.1 | 37.8  | 69.2      | 22.1  |  
| 44.5  | 39.3  | 45.1      | 10.4  |  
| 17.2  | 45.9  | 69.3      | 12.0  |  
| 151.5 | 41.3  | 58.5      | 16.5  |  
| 180.8 | 10.8  | 58.4      | 17.9  |  

---

## Steps  

### 1. Import Libraries  
The following Python libraries are used in this project:  
- `pandas`, `numpy`: For data manipulation.  
- `matplotlib.pyplot`, `seaborn`: For data visualization.  
- `scikit-learn`: For machine learning and evaluation.  
- `statsmodels`: For statistical analysis.  

---

### 2. Load the Dataset  
The dataset is loaded using the following command:  
```python
import pandas as pd  
df = pd.read_csv('advertising.csv')  
```  

---

### 3. Exploratory Data Analysis (EDA)  
Visualized the relationship between TV advertising spend and sales:  
```python
import matplotlib.pyplot as plt  

plt.figure(figsize=(12, 8))  
plt.scatter(df['TV'], df['Sales'])  
plt.xlabel('TV Advertising Budget')  
plt.ylabel('Sales')  
plt.title('TV Advertising vs Sales')  
plt.show()  
```  

---

### 4. Data Preparation  
- Features (X) and target (y) are defined as:  
  ```python
  X = df.iloc[:, 0:1]  # TV column  
  y = df.iloc[:, -1]   # Sales column  
  ```  

- The dataset is split into training and testing sets:  
  ```python
  from sklearn.model_selection import train_test_split  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)  
  ```  

---

### 5. Model Training  
A linear regression model is built and trained using `scikit-learn`:  
```python
from sklearn.linear_model import LinearRegression  

lr = LinearRegression()  
lr.fit(X_train, y_train)  
```  

---

### 6. Model Evaluation  
- The model's performance is evaluated using:  
  - Mean Squared Error (MSE).  
  - Root Mean Squared Error (RMSE).  
  - R-squared.  

- Key performance metrics:  
  ```python
  from sklearn.metrics import mean_squared_error  
  from math import sqrt  

  predictions = lr.predict(X_test)  
  rmse = sqrt(mean_squared_error(y_test, predictions))  
  print(f"RMSE: {rmse}")  
  ```  

- Regression equation:  
  ```python
  print(f"The linear model is: Y = {lr.intercept_:.5f} + {lr.coef_[0]:.5f}X")  
  ```  

---

### 7. Statistical Analysis  
Detailed statistical insights are obtained using `statsmodels`:  
```python
import statsmodels.api as sm  

X_train_sm = sm.add_constant(X_train)  
lr_sm = sm.OLS(y_train, X_train_sm).fit()  
print(lr_sm.summary())  
```  

---

### 8. Error Analysis  
Residuals are analyzed to verify model assumptions:  
```python
res = y_train - lr_sm.predict(X_train_sm)  

import seaborn as sns  

sns.histplot(res, bins=15, kde=True)  
plt.title('Residual Distribution')  
plt.xlabel('Error Terms')  
plt.show()  
```  

---

## Results  

- **Regression Equation:**  
  Y = 6.94868 + 0.05455X  

- **Key Metrics:**  
  - RMSE: 2.019  
  - R-squared: 0.816  

The model explains approximately 81.6% of the variance in sales based on TV advertising spend.  

---

## Conclusion  
- There is a significant positive relationship between TV advertising and sales.  
- The linear regression model is effective in predicting sales based on TV advertising budgets.  

---

## References  
- Kaggle: [Advertising Dataset](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input).  
- Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`.  

--- 

