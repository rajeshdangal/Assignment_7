import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("suv.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

entropy_model = DecisionTreeClassifier(criterion='entropy', random_state=20)
entropy_model.fit(X_train, y_train)

y_pred_entropy = entropy_model.predict(X_test)

print("Performance of Decision Tree (Entropy Criterion):")
print(confusion_matrix(y_test, y_pred_entropy))
print(classification_report(y_test, y_pred_entropy))

gini_model = DecisionTreeClassifier(criterion='gini', random_state=20)
gini_model.fit(X_train, y_train)

y_pred_gini = gini_model.predict(X_test)

print("Performance of Decision Tree (Gini Criterion):")
print(confusion_matrix(y_test, y_pred_gini))
print(classification_report(y_test, y_pred_gini))

print("Model Comparison:")
print("- The entropy model focuses on reducing uncertainty in the data, which might result in a more balanced tree.")
print("- The gini model is simpler and faster but can sometimes create slightly different decision boundaries.")
print("- Both methods can perform well; it's best to compare accuracy and precision to decide which suits this dataset better.")
