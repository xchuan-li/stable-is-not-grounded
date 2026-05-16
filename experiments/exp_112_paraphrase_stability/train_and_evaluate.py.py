import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
original = pd.read_csv("data/original.csv")
paraphrase = pd.read_csv("data/paraphrase.csv")

# training data
X_train = original["text"]
y_train = original["label"]

# vectorization
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

# model
model = LogisticRegression()

# train
model.fit(X_train_vec, y_train)

# evaluate on original
X_original_vec = vectorizer.transform(original["text"])
original_preds = model.predict(X_original_vec)

original_acc = accuracy_score(original["label"], original_preds)

# evaluate on paraphrase
X_para_vec = vectorizer.transform(paraphrase["text"])
para_preds = model.predict(X_para_vec)

para_acc = accuracy_score(paraphrase["label"], para_preds)

# consistency
consistency = (original_preds == para_preds).mean()

# print results
print("Original Accuracy:", original_acc)
print("Paraphrase Accuracy:", para_acc)
print("Consistency Rate:", consistency)

# detailed comparison
results = pd.DataFrame({
    "id": original["id"],
    "original_text": original["text"],
    "paraphrase_text": paraphrase["text"],
    "original_pred": original_preds,
    "paraphrase_pred": para_preds,
    "label": original["label"]
})

results["consistent"] = (
    results["original_pred"] == results["paraphrase_pred"]
)

print(results)

# save results
results.to_csv("data/results.csv", index=False)