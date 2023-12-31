import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Admission_Predict.csv")

df.head()
df.shape
df.columns

#for performing binarization we've to import it first
from sklearn.preprocessing import Binarizer

bi = Binarizer(threshold =0.75)

df['Chance of Admit '] = bi.fit_transform(df[['Chance of Admit ']])
df.head()

#now we've to divide data into x and y .. as y is our target variable

x = df.drop('Chance of Admit ',axis=1);
y = df['Chance of Admit ']

x.shape
y.shape
y = y.astype('int')
y

import seaborn as sns
sns.countplot(x = y)
y.value_counts()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25)

x_train.shape
x_test.shape

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)

classifier.fit(x_train,y_train)

DecisionTreeClassifier(random_state=0)
DecisionTreeClassifier(random_state=0)

y_pred =  classifier.predict(x_test)
y_pred

result = pd.DataFrame({
    'actual':y_test,
    'predicted':y_pred})

result

from sklearn.metrics import classification_report,ConfusionMatrixDisplay, accuracy_score
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
print(classification_report(y_test,y_pred))
new = [[1,337,118,4,4.5,4.5,9.65,1]]
classifier.predict(new)[0]

from sklearn.tree import plot_tree
plot_tree(classifier,);
plt.figure(figsize=(12,12))
plot_tree(classifier,fontsize=8,filled=True,rounded=True);