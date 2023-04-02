import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Features import FEATURE_LIST
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../data/wine_data_train.csv')
df = df[FEATURE_LIST]

df_test = pd.read_csv('../data/wine_data_test.csv')
df_test = df_test[FEATURE_LIST]

X_train, X_val, y_train, y_val = train_test_split(df.drop("quality",axis=1), df[["quality"]], test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_val)
MAE = mean_absolute_error(y_pred, y_val)
MSE = mean_squared_error(y_pred,y_val)
r_2 = reg.score(X_val, y_val)

# Perform of test.
X_test = df_test.drop("quality",axis=1)
y_test = df_test["quality"]
y_pred_test = reg.predict(X_test)
MAE_t = mean_absolute_error(y_pred_test, y_test)
MSE_t = mean_squared_error(y_pred_test,y_test)
r_2_t = reg.score(X_test, y_test)

message = (f'Validation R^2: {r_2}\n' + f'Validation MAE: {MAE}\n' + f'Validation MSE: {MSE}\n' 
           + f'Test R^2: {r_2_t}\n' + f'Test MAE: {MAE_t}\n' + f'Test MSE: {MSE_t}')
print(message)

with open("results/linear_regression_results.txt", "w") as file:
    file.write(message)