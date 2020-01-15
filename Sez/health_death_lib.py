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
from sklearn.preprocessing import PolynomialFeatures
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError


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
#High correlation
    def correlated(train_set):
        attributes = pd.DataFrame(train_set.corrwith(train_set['Life Expectancy']).abs() > 0.3)
        attributes.reset_index(inplace=True)
        attributes.columns = ['Attribute','Correlation']
        list_attr = attributes.loc[attributes['Correlation'] == True, 'Attribute']
        
        df_att = train_set[list(list_attr)]
        fig = plt.figure(figsize=(16,12))
        sns.heatmap(df_att.corr().abs(), annot=True)
        
        corr = df_att.corr() < 0.75
        corr.reset_index(drop=False,inplace=True)
        corr.rename(columns={'index': 'Attribute'},inplace=True)
        return corr, df_att
#Multicollinearity check
    def multicollinear(df):
        for i in df.columns:
            for j in list(range(1,len(df))):
                if df[i][j] == False:
                    if i != df.Attribute[j]:
                        print(f'High chance of multicollinearity between: {i} and {df.Attribute[j]}\n')
                    else:
                        pass
                else:
                    pass
#Scatter plots
    def scatter_plots(df,target):
        for i in df.columns:
            fig = plt.figure(figsize=(10,6))
            sns.scatterplot(x = df[i],y=target)
            plt.xlabel(i)
            plt.ylabel('Life Expectancy')
            plt.show()

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
    
class Model_Selection():
#Linear model    
    def lin_model(predictor, target):
        lin_reg = LinearRegression()
        c_val = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(lin_reg, predictor, target, scoring='neg_mean_squared_error', cv=c_val)
        r2_score = np.mean(cross_val_score(lin_reg, predictor, target, scoring='r2', cv=c_val))
        rmse_scores = np.sqrt(-scores)
        return rmse_scores, r2_score
#Display the model results
    def display_scores(model_name, rmse_scores, r2):
        print(model_name)
        print('Scores: ', rmse_scores)
        print('Mean: ', rmse_scores.mean())
        print('Standard Deviation: ',rmse_scores.std())
        print('r2: ', r2)
#Decision Tree model        
    def decision_tree_model(predictor, target):
        tree_reg = DecisionTreeRegressor()
        c_val = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores_tree = cross_val_score(tree_reg, predictor, target, scoring='neg_mean_squared_error', cv=c_val)
        tree_r2 = np.mean(cross_val_score(tree_reg, predictor, target, scoring='r2', cv=c_val))
        rmse_scores = np.sqrt(-rmse_scores_tree)
        return rmse_scores, tree_r2
#Lasso model
    def lasso_model(predictor,target,alpha_value):
        lasso = Lasso(alpha=alpha_value)
        c_val = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_lasso = cross_val_score(lasso, predictor, target, scoring='neg_mean_squared_error', cv=c_val)
        lasso_r2 = np.mean(cross_val_score(lasso, predictor, target, scoring='r2', cv=c_val))
        lasso_rmse = np.sqrt(-scores_lasso)
        return lasso_rmse, lasso_r2
#Random Forest Model
    def random_forest_model(predictor,target):
        forest_reg = RandomForestRegressor()
        c_val = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_forest = cross_val_score(forest_reg, predictor, target, scoring='neg_mean_squared_error', cv=10)
        forest_r2 = np.mean(cross_val_score(forest_reg, predictor, target, scoring='r2', cv=c_val))
        forest_rmse = np.sqrt(-scores_forest)
        return forest_rmse, forest_r2
#OLS
    def ols(predictor, target):
        preds = sm.add_constant(predictor)
        ols_model = sm.OLS(target,preds)
        ols_result = ols_model.fit()
        ols_result.summary()
        return ols_result
#OLS residual check distplot, qqplot, scatter    
    def ols_residual_check(ols_result, target, alpha):
        k2_resid, p_resid = stats.normaltest(ols_result.resid)
        if p_resid < alpha:  # null hypothesis: x comes from a normal distribution
            print("Residual is not normal",p_resid)
        else:
            print("Residual is normal")
        sns.distplot(ols_result.resid)
        plt.show()
        plt.scatter(ols_result.resid,target)
        plt.show()
        sm.qqplot(ols_result.resid, line='45')
#Add Polynomial Features
    def polynomial(degree,predictor, target):
        poly = PolynomialFeatures(degree)
        lin_reg = LinearRegression()
        df1_poly = poly.fit_transform(predictor)
        c_val = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_poly = cross_val_score(lin_reg, df1_poly, target, scoring='neg_mean_squared_error', cv=c_val)
        poly_r2 = np.mean(cross_val_score(lin_reg, df1_poly, target, scoring='r2', cv=c_val))
        poly_rmse = np.sqrt(-scores_poly)
        return poly_rmse, poly_r2
        
class Test():
#Test Data    
    def test_data(test_set, predictor):
        y_test = test_set['Life Expectancy']
        x_test = test_set[list(predictor.columns)]
        return x_test, y_test
#Visualize
    def residual_plot(lin_model,x_train, y_train, x_test, y_test):
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        visualizer = ResidualsPlot(lin_model, ax=ax)

        fig = plt.figure(figsize=(16,12))
        visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(x_test, y_test)  # Evaluate the model on the test data
        visualizer.show()
        
    def prediction_error_plot(lin_model,x_train, y_train, x_test, y_test):
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(111)
        visualizer_pred_err = PredictionError(lin_model, ax=ax1)

        visualizer_pred_err.fit(x_train, y_train)  # Fit the training data to the visualizer
        visualizer_pred_err.score(x_test, y_test)  # Evaluate the model on the test data
        visualizer_pred_err.show()
