#Import Libraries:

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import pandas as pd

#This function identify if there is a duplicated data to visualize the distribution of features
def check_duplicated(df_sort):

    #check for duplicated
    duplicated_rows = df_sort[df_sort.duplicated(keep=False)]

    if duplicated_rows.empty:
        message = "No duplicate rows found."
        dup = None
    else:
        message = "Duplicate rows found:"
        dup = duplicated_rows.sort_index()
    
    return (message,dup)

#This function return a histogram 
def data_hist(df,num_data):

    #Suppress the FutureWarning from seaborn
    warnings.simplefilter(action='ignore', category=FutureWarning)

    #Check how much data are in num_data to organize plots
    n_data = len(num_data)

    #simple visualization for 1D Array:
    if n_data ==1:
        sns.histplot(x=num_data[0],data=df)
        plt.title(num_data[0])
    elif n_data == 2 or n_data == 3:
        fig, ax = plt.subplots(1,n_data, figsize=(10,7))
        i = 0
        for _ in num_data:
            sns.histplot(x=_,data=df,kde=True,ax=ax[i])
            ax[i].set_title(_)
            i = i+1
    else:
        n_rows = n_data//3

    #use subplots to plot all data at once:
        fig, ax = plt.subplots(n_rows,3,figsize=(15,20))

        row = 0
        column = 0

        for _ in num_data:
            if row < n_rows:
                if column < 3:
                    sns.histplot(x=_,data=df,ax=ax[row,column])
                    ax[row,column].set_xlabel(_)
                    column = column+1
                else:
                    row = row+1
                    column = 0
                    sns.histplot(x=_,data=df,ax=ax[row,column])
                    ax[row,column].set_xlabel(_)
                    column = column+1
            else:
                break
    
    plt.tight_layout()

    return plt.show()

#This function finds outliers in a specified column of a DataFrame using the IQR method.
def find_iqr_outliers(df, column_list):

    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Return the rows where the value is an outlier
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = outliers[column]
        if outliers.count() == 0:
            print("Column ",column," without outliers.")
        else:
            print("Column ",column," : ",outliers.count()," outliers from ",outliers.min()," to ",outliers.max(),". Mean: ",outliers.mean())

def frequency(df,colum_list,group = False):

    if group is False:
        for column in colum_list:
            print(column,":")
            
            counts = df[column].value_counts()
            percentages = df[column].value_counts(normalize=True) * 100

            for index, count in counts.items():
                percentage = percentages[index]
                    
                print(f'- {index}:{count} ({percentage:.2f}%)', end=",\n")
            
    else:
        for column in colum_list:

            others = []

            percentages = df[column].value_counts(normalize=True) * 100

            for index, percent in percentages.items():
                percent = percentages[index]
                if percent <= 5:
                    others.append(index)
            if len(others) > 1:
                df[column].replace(others,-1,inplace=True)
                others_percentage = df[column].value_counts(normalize=True)*100
                for index, percent in others_percentage.items():
                    if index == -1:
                        print(f'{column} grouped: {percent:.2f}%')
        return df

#This function returns evaluations for each model and their predictions
def evaluations(model,y_pred,y_test):

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Use 'weighted' average for multiclass/imbalanced binary classification
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store metrics in a dictionary
    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    # Print the four core metrics
    print(f"Overall Accuracy for {model}: {metrics['Accuracy']:.4f}")
    print(f"Weighted Precision for {model}: {metrics['Precision']:.4f}")
    print(f"Weighted Recall for {model}: {metrics['Recall']:.4f}")
    print(f"Weighted F1-Score for {model}: {metrics['F1-Score']:.4f}")

    # Print Confusion Matrix
    print("\n" + "-" * 20 + " Confusion Matrix " + "-" * 20)
    cm = confusion_matrix(y_test, y_pred)
    # Convert CM to DataFrame for better readability, if feasible

    cm_df = pd.DataFrame(cm, columns=["Actual 0","Actual 1"])
    print("Predicted:")
    print(cm_df)
