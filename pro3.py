import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv("python_learning_exam_performance.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
df = df.drop(columns=["country","student_id"])
print(df.columns)

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["age"],y=df["hours_spent_learning_per_week"])
plt.title("Age vs Hours Spent Learning Per Week")
plt.show()
plt.figure(figsize=(8,6))
sns.histplot(df["practice_problems_solved"])
plt.title("Practice Problems Solved Distribution")
plt.show()
plt.figure(figsize=(8,6))
sns.histplot(df["final_exam_score"])
plt.title("Final Exam Score Distribution")
plt.show()
plt.figure(figsize=(8,6))
sns.boxplot(x=df["passed_exam"],y=df["final_exam_score"])
plt.title("Final Exam Score by Exam Passing Status")
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["practice_problems_solved"],y=df["final_exam_score"])
plt.title("Practice Problems Solved vs Final Exam Score")
plt.show()
num = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(8,6))
sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Numeric Correlation Matrix')
plt.show()

df.dropna(inplace=True)
print(df.isnull().sum())
print(df.head())
print(df["prior_programming_experience"].unique())
df["prior_programming_experience"]=df["prior_programming_experience"].replace("Beginner",0)
df["prior_programming_experience"]=df["prior_programming_experience"].replace("Intermediate",1)
df["prior_programming_experience"]=df["prior_programming_experience"].replace("Advanced",2)
df["prior_programming_experience"]=df["prior_programming_experience"].replace("None",np.nan)
df.dropna(subset=["prior_programming_experience"])
#df["prior_programming_experience"]=df["prior_programming_experience"].map({"Beginner":0,"Intermediate":1,"Advanced":2})
print(df.head())

X=df.drop(columns=["passed_exam"])
y=df["passed_exam"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
ac=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
c=classification_report(y_test,y_pred)

print("Logistic Regression Results:")
print("Accuracy Score:", ac*100)
print("Confusion Matrix:\n", cm)
print("Classification report:\n", c)

model1=DecisionTreeClassifier(random_state=42)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
ac1=accuracy_score(y_test,y_pred1)
cm1=confusion_matrix(y_test,y_pred1)
c1=classification_report(y_test,y_pred1)

print("Decision Tree Classifier Results:")
print("Accuracy Score:", ac1*100)
print("Confusion Matrix:\n", cm1)
print("Classification report:\n", c1)

model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)
cm2 = confusion_matrix(y_test, y_pred2)
c2 = classification_report(y_test, y_pred2)

print("Random Forest Classifier Results:")
print("Accuracy Score:", ac2*100)
print("Confusion Matrix:\n", cm2)
print("Classification report:\n", c2)

print('\nSummary: Logistic, DecisionTree, RandomForest accuracies (percent):')
print(f"Logistic: {ac*100:.2f} | DecisionTree: {ac1*100:.2f} | RandomForest: {ac2*100:.2f}")

