import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class Eda():
    
    def get_df(sheet_name,xcl):
        df = pd.read_excel(xcl, sheet_name=sheet_name)
        df.columns = list(df.iloc[0])
        df.drop([0],inplace=True)
        return df
#get neccessary columns from added measures dataframe 
    def get_add_df(df):
        life_exp = df[['FIPS','State','County','Life Expectancy', '% Frequent Physical Distress',
                       '% Frequent Mental Distress', '% Diabetic', '% Food Insecure',
                        '% Insufficient Sleep', 'Household Income', '% Homeowners',
                       '% Severe Housing Cost Burden']]
        return life_exp
#get neccessary columns from ranked measures dataframe    
    def get_ranked_df(df):
        factors = df[['FIPS','% Fair/Poor',
                         '% LBW', '% Smokers', '% Obese', 'Food Environment Index',
                         '% Physically Inactive', '% With Access', '% Excessive Drinking',
                         '% Alcohol-Impaired','Chlamydia Rate', 'Teen Birth Rate', '% Uninsured',
                          'Dentist Rate', 'Preventable Hosp. Rate', '% Screened', '% Vaccinated',
                         'Graduation Rate', '% Some College', '% Unemployed', '% Children in Poverty',
                          'Income Ratio', '% Single-Parent Households', 'Association Rate', 'Average Daily PM2.5',
                         'Presence of violation', '% Severe Housing Problems','% Drive Alone',
                         '% Long Commute - Drives Alone']]
        return factors
#Merging dataframes and dropping null values    
    def merge_dropna(df1,df2):
        df_merged = df1.merge(df2, how='left',on='FIPS')
        df_merged.dropna(subset=['Life Expectancy'], inplace=True)
        df_final = df_merged.dropna()
        return df_final
#Have a dataframe with numerical values to use in our model   
    def get_num_df(df):
        num_col = []
        for i in df.columns:
            if (type(df[i][0]) != str):
                num_col.append(i)
        num = df[num_col].astype('float64')
        return num 
    #7 highly correlated columns    
    def get_df_seven(train_set):
        corr = train_set.corrwith(train_set['Life Expectancy']).abs().sort_values(ascending=False).head(8)
        df_seven = train_set[list(corr[1:].index)]
        fig = plt.figure(figsize=(12,8))
        sns.heatmap(df_seven.corr().abs(), annot=True)
        return df_seven

class Prep():
#Remove Outliers    
    def remove_outliers(train_set):
        for i in train_set.columns:
            train_set = train_set.loc[train_set[i] < np.percentile(train_set[i],99.5)]
            train_set = train_set.loc[train_set[i] > np.percentile(train_set[i],0.5)]
        train_set = train_set[(np.abs(stats.zscore(train_set)) < 3.2).all(axis=1)]
        train_set.reset_index(drop=True,inplace=True)
        return train_set
#Normality test
    def normality_check(df):
        alpha = 0.05
        for i in df.columns:
            fig = plt.figure(figsize=(12,8))
            sns.distplot(df[i], label=i, hist=False)
            k2, p = stats.normaltest(df[i])
            if p < alpha:  # null hypothesis: x comes from a normal distribution
                 print("%s is not normal"%i,p)
            else:
                 print("%s is normal"%i)
#Get scaled df
    def scale(predictors):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(predictors)
        df_scaled = pd.DataFrame(scaled)
        df_scaled.columns = predictors.columns
        return df_scaled
    
    