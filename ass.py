import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def load_data(file_path):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.readlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode the file {file_path} with any of the attempted encodings.")

try:
    pos_sentences = load_data('rt-polaritydata\\rt-polaritydata\\rt-polarity.pos')
    neg_sentences = load_data('rt-polaritydata\\rt-polaritydata\\rt-polarity.neg')
except ValueError as e:
    print(f"Error loading data: {e}")
    exit(1)

train_pos = pos_sentences[:4000]
train_neg = neg_sentences[:4000]

val_pos = pos_sentences[4000:4500]
val_neg = neg_sentences[4000:4500]

test_pos = pos_sentences[4500:5331]
test_neg = neg_sentences[4500:5331]

train_sentences = train_pos + train_neg
train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

val_sentences = val_pos + val_neg
val_labels = np.array([1] * len(val_pos) + [0] * len(val_neg))

test_sentences = test_pos + test_neg
test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_sentences)
X_val = vectorizer.transform(val_sentences)
X_test = vectorizer.transform(test_sentences)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary')

accuracy = accuracy_score(test_labels, y_pred)

print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
