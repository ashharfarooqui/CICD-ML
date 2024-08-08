import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# Loading the data from csv file 
df = pd.read_csv("Data/drug200.csv")
df = df.sample(frac=1)
df.head(5)

# This section is for train test split data
X = df.drop("Drug", axis=1).values
y = df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=150
)

# Creating a pipeline code here
cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=150)),
    ]
)

# Training the pipeline
pipe.fit(X_train, y_train)

# model evaluation is here
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Here is confusion matrix plot 
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(cmap=plt.cm.Greens)
plt.savefig("Results/model_results.png", dpi=100)


# Assuming you have your pipeline defined as 'pipe'
# save the model file
joblib.dump(pipe, "Model/drug_pipeline.joblib")