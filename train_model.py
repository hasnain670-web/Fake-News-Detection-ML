import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import( accuracy_score
                            ,precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            classification_report,
                            ConfusionMatrixDisplay
                            )
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
fake=pd.read_csv("dataset/Fake.csv")
true=pd.read_csv("dataset/True.csv")

print(fake.head())
print(true.head())
print(fake.shape)
print(true.shape)

print(fake.columns)
print(true.columns)

print(fake.info())
print(true.info())

print(fake.isnull().sum())
print(fake.isnull().sum())

true["label"]=1
fake["label"]=0

df=pd.concat([fake,true], ignore_index=True)
df=df.sample(frac=1,random_state=42)
print(df.head())
print(df.shape)

sns.countplot(x="label",data=df)
plt.title("Fake Vs Real News")
plt.show()

df.to_csv("dataset/news.csv",index=False)

nltk.download("stopwords")
stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()

def clean_text(text):
    text=text.lower()
    text=re.sub(f"[{string.punctuation}]","",text)
    text=re.sub(r"\d+","",text)
    text=re.sub(r"\s+","",text)
    words=text.split()
    words=[word for word in  words if word not in stop_words ]
    words=[stemmer.stem(word) for word in words]
    return "".join(words)

df["content"]=df["title"]+" "+df["text"]
df["content"]=df["content"].apply(clean_text)
print(df[["content","label"]].head())
df.to_csv("dataset/news_clean.csv",index=False)

x=df["content"]
y=df["label"]

vectorizer=TfidfVectorizer(max_features=5000)
x=vectorizer.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LogisticRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
print(lr_pred)
lr_acc=accuracy_score(y_test,lr_pred)
print("Logistic Regression Accuracy:",lr_acc)

nb=MultinomialNB()
nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)
nb_acc=accuracy_score(y_test,nb_pred)
print("Naive Bayes Accuracy:",nb_acc)

pac=PassiveAggressiveClassifier(max_iter=1000)
pac.fit(x_train,y_train)
pac_pred=pac.predict(x_test)
pac_acc=accuracy_score(y_test,pac_pred)
print("Passive Aggressive Accuracy:",pac_acc)

print("\nModel Comparsion")
print("=============")
print("Logistic Regression :",lr_acc)
print("Naive Bayes :",nb_acc)
print("Passive Aggressive :",pac_acc)

best_model=pac
joblib.dump(best_model,"models/fake_news_mode.pkl")
print("Model save successfully!")

joblib.dump(vectorizer,"models/vectorizer.pkl")
print("Vectorizer Saved!")

accuracy=accuracy_score(y_test,lr_pred)
print("Accuracy:",accuracy)

precision=precision_score(y_test,lr_pred)
print("Precision:",precision)

recall=recall_score(y_test,lr_pred)
print("Recall:",recall)

f1=f1_score(y_test,lr_pred)
print("F1 Score:",f1)

print(classification_report(y_test,lr_pred))

cm=confusion_matrix(y_test,lr_pred)
print(cm)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake","Real"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

def evaluate_model(name,y_true,y_pred):
    print("=" *50)
    print(name)
    print("="*50)
    print("Accuracy :",accuracy_score(y_true,y_pred))
    print("Precision :",precision_score(y_true,y_pred))
    print("Recall :",recall_score(y_true,y_pred))
    print("F1 Score :",f1_score(y_true,y_pred))
    print("\nClassification Report")
    print(classification_report(y_true,y_pred))

evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred
)
evaluate_model(
    "Naive Bayes",
    y_test,
    nb_pred
)

evaluate_model(
    "Passive Aggressive",
    y_test,
    pac_pred
)

results={
    "Logistic Regression": accuracy_score(y_test,lr_pred),
    "Naive Bayes": accuracy_score(y_test,lr_pred),
    "Passive Aggressive":accuracy_score(y_test,lr_pred)
}
print("\nModel Comparison")

for model,score in results.item():
    print(f"{model}:{score:.4f}")

best_model=pac
joblib.dump(
    best_model,
    "models/fake_news_mode.pkl"
)
joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)
print("Best Model Saved Successfully!")

