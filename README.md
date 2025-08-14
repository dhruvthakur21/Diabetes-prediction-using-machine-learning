# Diabetes Prediction Project

This project uses **Logistic Regression** to predict whether a patient has diabetes based on diagnostic measurements. The dataset used is the Pima Indians Diabetes Dataset.

## Dataset

- **Pregnancies**: Number of times pregnant   
- **Glucose**: Plasma glucose concentration  
- **BloodPressure**: Diastolic blood pressure (mm Hg)  
- **SkinThickness**: Triceps skinfold thickness (mm)  
- **Insulin**: 2-Hour serum insulin (mu U/ml)  
- **BMI**: Body mass index (weight in kg/(height in m)^2)  
- **DiabetesPedigreeFunction**: Diabetes pedigree function   
- **Age**: Age in years  
- **Outcome**: Class variable (0 = no diabetes, 1 = diabetes)  

## Project Steps

1. **Data Loading and Exploration**  
   - Load the dataset using Pandas.  
   - Check for missing values and basic statistics.  
   - Visualize correlations using a heatmap.

2. **Data Preprocessing**  
   - Split dataset into features (`X`) and labels (`y`).  
   - Standardize features using `StandardScaler`.

3. **Model Training**  
   - Train a **Logistic Regression** model on the training set.

4. **Prediction and Evaluation**  
   - Make predictions on the test set.  
   - Evaluate model performance using:  
     - Accuracy Score  
     - Confusion Matrix  
     - Classification Report

5. **Visualization**  
   - Plot confusion matrix and correlation heatmap using Seaborn and Matplotlib.

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

## Model Performance

The Logistic Regression model achieves an **accuracy of around 78–80%** (may vary slightly depending on the train/test split).  

## How to Run

1. Clone the repository.  
2. Install the required packages:  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
