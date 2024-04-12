import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import IsolationForest

df = pd.read_csv('forestfires.csv')
df_10 = df.head(10)

# print(df_10)
# print(df.count)

# ======------------======

# pd.set_option("display.max_columns", 31)
# y_count = df.size_category.value_counts().reset_index().rename(columns={'count':'counts'})
# plt.figure(figsize=(8, 8))
# plt.pie(y_count.count, labels=y_count['size_category'], autopct='%1.2f%%', explode=(0, 0.02))
# plt.show()

# ======------------======

month_df = df.groupby(['size_category', 'month']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
month_df.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='month', y='count', hue='size_category', data=month_df)
plt.title("Num of fires in each month", fontsize=17, y=1.02)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

day_df = df.groupby(['size_category', 'day']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
day_df.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='day', y='count', hue='size_category', data=day_df)
plt.title("Num of fires in each day(Monday to Sunday)", fontsize=17, y=1.02)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# ======------------======

labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
# print(df['size_category'])

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= RAIN

rain_df = df.groupby(['size_category', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(12, 6))
rain_df['size_category'] = rain_df['size_category'].astype(str)
sns.barplot(x='rain', y='count', hue='size_category', data=rain_df)
plt.title("Rainfall level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('Rain', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= RH

rh_df = df.groupby(['size_category', 'RH']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
rh_df['size_category'] = rh_df['size_category'].astype(str)
sns.barplot(x='RH', y='count', hue='size_category', data=rh_df)
plt.title("RH level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('RH', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= FFMC

ffmc_df = df.groupby(['size_category', 'FFMC']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
ffmc_df['size_category'] = ffmc_df['size_category'].astype(str)
sns.barplot(x='FFMC', y='count', hue='size_category', data=ffmc_df)
plt.title("FFMC level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('FFMC', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= DMC

dmc_df = df.groupby(['size_category', 'DMC']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
dmc_df['size_category'] = dmc_df['size_category'].astype(str)
sns.barplot(x='DMC', y='count', hue='size_category', data=dmc_df)
plt.title("DMC level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('DMC', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= DC

dc_df = df.groupby(['size_category', 'DC']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
dc_df['size_category'] = dc_df['size_category'].astype(str)
sns.barplot(x='DC', y='count', hue='size_category', data=dc_df)
plt.title("DC level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('DC', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= ISI

isi_df = df.groupby(['size_category', 'ISI']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
isi_df['size_category'] = isi_df['size_category'].astype(str)
sns.barplot(x='ISI', y='count', hue='size_category', data=isi_df)
plt.title("ISI level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('ISI', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= TEMP

temp_df = df.groupby(['size_category', 'temp']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
temp_df['size_category'] = temp_df['size_category'].astype(str)
sns.barplot(x='temp', y='count', hue='size_category', data=temp_df)
plt.title("Temp level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('Temp', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= WIND

wind_df = df.groupby(['size_category', 'wind']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
wind_df['size_category'] = wind_df['size_category'].astype(str)
sns.barplot(x='wind', y='count', hue='size_category', data=wind_df)
plt.title("Wind level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('Wind', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-= AREA

area_df = df.groupby(['size_category', 'area']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

plt.figure(figsize=(40, 6))
area_df['size_category'] = area_df['size_category'].astype(str)
sns.barplot(x='area', y='count', hue='size_category', data=area_df)
plt.title("Area level in diff category of forest", y=1.02, fontsize=17)
plt.xlabel('Area', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.show()

# ======------------======

df.drop(['month', 'day', 'monthjan', 'daymon'], axis=1, inplace=True)
pd.set_option("display.max_columns", 27)
df.head()

data1 = df.copy()

# training the model
clf = IsolationForest(random_state=10, contamination=.01)
clf.fit(data1)
data1['anomoly'] = clf.predict(data1.iloc[:,0:27])
outliers = data1[data1['anomoly']==-1]
# print(outliers)

outliers.index
df.drop([281, 299, 379, 463, 464, 469], axis=0, inplace=True)
# print(df.shape)

x = df.drop('size_category', axis=1)
y = df['size_category']

norm = MinMaxScaler()
std = StandardScaler()

x_norm = pd.DataFrame(norm.fit_transform(x), columns=x.columns)
x_std = pd.DataFrame(std.fit_transform(x), columns=x.columns)

# print(x_std.head())

# ======------------======

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=.25)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# ======------------======

y_train = y_train.astype('int')
y_test = y_test.astype('int')

clf = SVC()
param_grids = [{'kernel': ['rbf'], 'C': [15,14,13,12,11,10,1,0.1,0.001]}]
grid = GridSearchCV(clf, param_grids, cv=20)
grid.fit(x_train, y_train)
# print(grid.best_score_, grid.best_params_)

# ======------------======

best_C = grid.best_params_['C']

print('RBF Kernel with gamma as scale')

rbf = SVC(gamma='scale')
param_grids = [{'kernel': ['rbf'], 'C': [best_C]}]
grid = GridSearchCV(rbf, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('RBF Kernel with gamma as auto')

rbf = SVC(gamma='auto')
param_grids = [{'kernel': ['rbf'], 'C': [best_C]}]
grid = GridSearchCV(rbf, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('Grid search for the kernel of Linear')

lin = SVC()
param_grids = [{'kernel': ['linear'], 'C': [best_C]}]
grid = GridSearchCV(lin, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('Grid search for the kernel of Polynomial')

poly = SVC()
param_grids = [{'kernel': ['poly'], 'C': [best_C]}]
grid = GridSearchCV(poly, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)


