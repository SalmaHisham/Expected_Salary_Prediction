############################################################ IMPORTANT LIBRARIES ############################################################
import numpy as np 
import pandas as pd 
#spliting data
from sklearn.model_selection import train_test_split
#models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
#adding constant 
import statsmodels.api as sm
#featurn selection
from sklearn.feature_selection import SelectKBest, f_regression
#saving the model
import pickle
########################################################### IMPORTANT FUNCTIONS ##############################################################
# read data and define their columns either numerical or categorical 
def read_data(path):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    num_cols = list(set(df.select_dtypes(exclude=[object]).columns) - set (['min_salary', 'max_salary']))
    cat_cols = list(set(df.select_dtypes(include=[object]).columns) - set(['Job Title', 'Job Description', 'Company Name', 'Competitors', 'Industry', 'Location', 'Headquarters']))  
    return df, num_cols, cat_cols

# Handle Null Values 
def handle_nulls(df, cat_cols):
    # categorical columns
    for column in cat_cols:
        df[column].fillna(df[column].mode()[0], inplace=True)
    # Numerical columns
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
    df["company_age"] = df["company_age"].fillna(df["company_age"].median())
    df["Founded"] = df["Founded"].fillna(df["Founded"].median())
    return df 

# Encode categorical data and merge them back
def Encode_Categorical_Cols(df, cat_cols, num_cols):
    dummy_encoded_df = pd.get_dummies(df[cat_cols], drop_first = True)
    df_dummy = pd.concat([df[num_cols], dummy_encoded_df], axis=1)
    return df_dummy

# Feature selection using f_regression
def feature_selection(X, y, K):
    select_feature = SelectKBest(f_regression, k=K).fit(X, y)
    X2 = select_feature.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def split_predictors(df_dummy):
    df_dummy = sm.add_constant(df_dummy)
    X = df_dummy.drop(["avg_salary"], axis = 1)
    y = df_dummy[["avg_salary"]]
    return X, y

def preprocessing(df,cat_cols, num_cols,  K = 22): 
    df = handle_nulls(df, cat_cols)
    df_dummy = Encode_Categorical_Cols(df, cat_cols, num_cols)
    X, y = split_predictors(df_dummy)
    X_train, X_test, y_train, y_test = feature_selection(X, y, K)
    return X_train, X_test, y_train, y_test

#Trying the model 
def Bagging_Regressor(X_train, X_test, y_train, y_test):
    # build the model
    bag_clf = BaggingRegressor(DecisionTreeRegressor(random_state=42), n_estimators=1000 , max_samples= 350, bootstrap=True)
    # fit the model
    bag_clf.fit(X_train, y_train.values.flatten())
    return bag_clf

#Saving the model
def save_model(MODEL):
    pickl = {'model': MODEL}
    pickle.dump( pickl, open( 'FlaskAPI/models/model_file' + ".p", "wb" ) )
    file_name = "FlaskAPI/models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model 


#################################################################### MAIN #################################################################### 
if __name__ == "__main__":
    df, num_cols, cat_cols = read_data("cleaned_data.csv")
    X_train, X_test, y_train, y_test = preprocessing(df,cat_cols, num_cols,  K = 22)
    bag_clf_model = Bagging_Regressor(X_train, X_test, y_train, y_test)
    model = save_model(bag_clf_model)
    # model.predict(X_test)
    


