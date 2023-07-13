import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv(r"C:\Users\prashant\Downloads\BODY_FAT\bodyfat.csv")
data.head()

data.columns

data = data[['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
       'Wrist','BodyFat']]

data.head()

data.describe(include='all')

data.shape

data.dtypes

for i in data.columns:
    print({i: data[i].unique()})

data.isnull().sum()

data.duplicated().sum()

# Assumption 1 : There should be no outliers

#for i in data.columns:
   # data.boxplot(column=i)
   # plt.show()

q1 = data['Density'].quantile(0.25)
q3 = data['Density'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Density_data_include = data.loc[(data['Density'] >= low) & (data['Density'] <= high)]
Density_data_exclude = data.loc[(data['Density'] < low) | (data['Density'] > high)]

Density_data_exclude.loc[Density_data_exclude['Density'] < low, 'Density'] = low
Density_data_exclude.loc[Density_data_exclude['Density'] > high, 'Density'] = high

data = pd.concat([Density_data_include, Density_data_exclude],axis=0)
data.shape

q1 = data['Weight'].quantile(0.25)
q3 = data['Weight'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Weight_data_include = data.loc[(data['Weight'] >= low) & (data['Weight'] <= high)]
Weight_data_exclude = data.loc[(data['Weight'] < low) | (data['Weight'] > high)]

Weight_data_exclude.loc[Weight_data_exclude['Weight'] < low, 'Weight'] = low
Weight_data_exclude.loc[Weight_data_exclude['Weight'] > high, 'Weight'] = high

data = pd.concat([Weight_data_include, Weight_data_exclude],axis=0)
data.shape

q1 = data['Height'].quantile(0.25)
q3 = data['Height'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Height_data_include = data.loc[(data['Height'] >= low) & (data['Height'] <= high)]
Height_data_exclude = data.loc[(data['Height'] < low) | (data['Height'] > high)]

Height_data_exclude.loc[Height_data_exclude['Height'] < low, 'Height'] = low
Height_data_exclude.loc[Height_data_exclude['Height'] > high, 'Height'] = high

data = pd.concat([Height_data_include, Height_data_exclude],axis=0)
data.shape

q1 = data['Neck'].quantile(0.25)
q3 = data['Neck'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Neck_data_include = data.loc[(data['Neck'] >= low) & (data['Neck'] <= high)]
Neck_data_exclude = data.loc[(data['Neck'] < low) | (data['Neck'] > high)]

Neck_data_exclude.loc[Neck_data_exclude['Neck'] < low, 'Neck'] = low
Neck_data_exclude.loc[Neck_data_exclude['Neck'] > high, 'Neck'] = high

data = pd.concat([Neck_data_include, Neck_data_exclude],axis=0)
data.shape

q1 = data['Chest'].quantile(0.25)
q3 = data['Chest'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Chest_data_include = data.loc[(data['Chest'] >= low) & (data['Chest'] <= high)]
Chest_data_exclude = data.loc[(data['Chest'] < low) | (data['Chest'] > high)]

Chest_data_exclude.loc[Chest_data_exclude['Chest'] < low, 'Chest'] = low
Chest_data_exclude.loc[Chest_data_exclude['Chest'] > high, 'Chest'] = high

data = pd.concat([Chest_data_include, Chest_data_exclude],axis=0)
data.shape

q1 = data['Abdomen'].quantile(0.25)
q3 = data['Abdomen'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Abdomen_data_include = data.loc[(data['Abdomen'] >= low) & (data['Abdomen'] <= high)]
Abdomen_data_exclude = data.loc[(data['Abdomen'] < low) | (data['Abdomen'] > high)]

Abdomen_data_exclude.loc[Abdomen_data_exclude['Abdomen'] < low, 'Abdomen'] = low
Abdomen_data_exclude.loc[Abdomen_data_exclude['Abdomen'] > high, 'Abdomen'] = high

data = pd.concat([Abdomen_data_include, Abdomen_data_exclude],axis=0)
data.shape

q1 = data['Hip'].quantile(0.25)
q3 = data['Hip'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Hip_data_include = data.loc[(data['Hip'] >= low) & (data['Hip'] <= high)]
Hip_data_exclude = data.loc[(data['Hip'] < low) | (data['Hip'] > high)]

Hip_data_exclude.loc[Hip_data_exclude['Hip'] < low, 'Hip'] = low
Hip_data_exclude.loc[Hip_data_exclude['Hip'] > high, 'Hip'] = high

data = pd.concat([Hip_data_include, Hip_data_exclude],axis=0)
data.shape

q1 = data['Thigh'].quantile(0.25)
q3 = data['Thigh'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Thigh_data_include = data.loc[(data['Thigh'] >= low) & (data['Thigh'] <= high)]
Thigh_data_exclude = data.loc[(data['Thigh'] < low) | (data['Thigh'] > high)]

Thigh_data_exclude.loc[Thigh_data_exclude['Thigh'] < low, 'Thigh'] = low
Thigh_data_exclude.loc[Thigh_data_exclude['Thigh'] > high, 'Thigh'] = high

data = pd.concat([Thigh_data_include, Thigh_data_exclude],axis=0)
data.shape

q1 = data['Knee'].quantile(0.25)
q3 = data['Knee'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Knee_data_include = data.loc[(data['Knee'] >= low) & (data['Knee'] <= high)]
Knee_data_exclude = data.loc[(data['Knee'] < low) | (data['Knee'] > high)]

Knee_data_exclude.loc[Knee_data_exclude['Knee'] < low, 'Knee'] = low
Knee_data_exclude.loc[Knee_data_exclude['Knee'] > high, 'Knee'] = high

data = pd.concat([Knee_data_include, Knee_data_exclude],axis=0)
data.shape

q1 = data['Ankle'].quantile(0.25)
q3 = data['Ankle'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Ankle_data_include = data.loc[(data['Ankle'] >= low) & (data['Ankle'] <= high)]
Ankle_data_exclude = data.loc[(data['Ankle'] < low) | (data['Ankle'] > high)]

Ankle_data_exclude.loc[Ankle_data_exclude['Ankle'] < low, 'Ankle'] = low
Ankle_data_exclude.loc[Ankle_data_exclude['Ankle'] > high, 'Ankle'] = high

data = pd.concat([Ankle_data_include, Ankle_data_exclude],axis=0)
data.shape

q1 = data['Biceps'].quantile(0.25)
q3 = data['Biceps'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Biceps_data_include = data.loc[(data['Biceps'] >= low) & (data['Biceps'] <= high)]
Biceps_data_exclude = data.loc[(data['Biceps'] < low) | (data['Biceps'] > high)]

Biceps_data_exclude.loc[Biceps_data_exclude['Biceps'] < low, 'Biceps'] = low
Biceps_data_exclude.loc[Biceps_data_exclude['Biceps'] > high, 'Biceps'] = high

data = pd.concat([Biceps_data_include, Biceps_data_exclude],axis=0)
data.shape

q1 = data['Forearm'].quantile(0.25)
q3 = data['Forearm'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Forearm_data_include = data.loc[(data['Forearm'] >= low) & (data['Forearm'] <= high)]
Forearm_data_exclude = data.loc[(data['Forearm'] < low) | (data['Forearm'] > high)]

Forearm_data_exclude.loc[Forearm_data_exclude['Forearm'] < low, 'Forearm'] = low
Forearm_data_exclude.loc[Forearm_data_exclude['Forearm'] > high, 'Forearm'] = high

data = pd.concat([Forearm_data_include, Forearm_data_exclude],axis=0)
data.shape

q1 = data['Wrist'].quantile(0.25)
q3 = data['Wrist'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

Wrist_data_include = data.loc[(data['Wrist'] >= low) & (data['Wrist'] <= high)]
Wrist_data_exclude = data.loc[(data['Wrist'] < low) | (data['Wrist'] > high)]

Wrist_data_exclude.loc[Wrist_data_exclude['Wrist'] < low, 'Wrist'] = low
Wrist_data_exclude.loc[Wrist_data_exclude['Wrist'] > high, 'Wrist'] = high

data = pd.concat([Wrist_data_include, Wrist_data_exclude],axis=0)
data.shape

# Assumption of Linearity : Every independent variable should have linear relationship with dependent variable

sns.pairplot(data,x_vars=['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
       'Wrist'], y_vars=['BodyFat'], kind='reg')

X = data[['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip',
       'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']]
Y = data['BodyFat']

print(X.shape)
print(Y.shape)

# Assumption of Normality : Dependent variable should follow an approximate normal distribution

sns.distplot(Y)

X.hist(bins=20)

from scipy.stats import skew
#finding the skewness in each variable
data_num_skew = X.apply(lambda i: skew(i.dropna()))  #dropping the null values using the lambda function and check the skewness
#subsetting the variables that are highly skewed
data_num_skewed = data_num_skew[(data_num_skew > .75) | (data_num_skew < -.75)]
 
print(data_num_skew)
print(data_num_skewed)
import numpy as np
# apply log + 1 transformation for all numeric features with skewness over .75
X[data_num_skewed.index] = np.log1p(X[data_num_skewed.index])

# Assumption of Multicollinearity : There should be no multicollinearity between the independent variables

corr_df = data.corr(method='pearson')
print(corr_df)

sns.heatmap(corr_df,vmax=1.0, vmin=-1.0, annot=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df['Features'] = X.columns
vif_df['VIF Factor'] = [vif(X.values, i)for i in range(X.shape[1])]
vif_df.round(2)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, Y_train)

print(lm.intercept_)
print(lm.coef_)

Y_pred = lm.predict(X_test)
print(Y_pred)

print(list(zip(Y_test, Y_pred)))

new_df = pd.DataFrame()
new_df = X_test.copy()

new_df['Actual BodyFat'] = Y_test
new_df['Predicted BodyFat'] = Y_pred
new_df = new_df.reset_index().drop('index',axis=1)
new_df

new_df['Actual BodyFat'].plot()
new_df['Predicted BodyFat'].plot()

#THIS WILL WORK ON UNSEEN DATA
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
 
r2=r2_score(Y_test,Y_pred)
print("R-squared:",r2)
 
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RMSE:",rmse)
 
adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)

filename = r'lm.sav'
pickle.dump(lm, open(filename,'wb'))
loaded_model = pickle.load(open(filename,'rb'))

filename = r"lm.sav"
pickle.dump(lm,open("body.pkl",'wb'))
model = pickle.load(open("body.pkl",'rb'))