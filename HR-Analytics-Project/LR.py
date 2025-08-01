import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy as sa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Connect to the PostgreSQL database
# Ensure you have the psycopg2 library installed: pip install psycopg2

conn = psycopg2.connect(
    host="localhost",
    dbname="HR_Analytics",
    user="postgres",
    password="prashant@1234"
)

query = "SELECT * FROM hr_data;"
print("Connected to the database successfully.")

# Read data from the PostgreSQL database into a pandas DataFrame
df = pd.read_sql(query, conn)

# Display the first few rows of the DataFrame
print(df.head())

# _------------------------------------------------------------------------------------------------------------

#  Step 2: Clean & Encode the Data

# Drop useless columns
df.drop(['employeecount', 'over18', 'standardhours', 'employeenumber'], axis=1, inplace=True)

# Encode binary columns
le = LabelEncoder()
df['attrition'] = le.fit_transform(df['attrition'])  # Target
df['gender'] = le.fit_transform(df['gender'])        # Male=1, Female=0
df['overtime'] = le.fit_transform(df['overtime'])    # Yes=1, No=0

# One-hot encode remaining categoricals
df = pd.get_dummies(df, columns=['businesstravel', 'department', 'educationfield', 'jobrole', 'maritalstatus'], drop_first=True)

# ---------------------------------------------------------------------------------------------------------------------------

#  Train-Test Split

X = df.drop('attrition', axis=1)
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Train a Machine Learning Model ---  Logistic Regression (Start simple)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# -----------------------------------------------------------------------------------------------------------------------
print("Non-numeric columns left:", X.select_dtypes(include='object').columns)

# Try Other Models 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", rf.score(X_test, y_test))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
print("XGBoost Accuracy:", xgb.score(X_test, y_test))

# --------------------------------------------------------------------

# Feature Importance Visualization
import matplotlib.pyplot as plt
import seaborn as sns

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

# auto-fix for future?
assert all(df.dtypes != 'object'), "Some columns are still non-numeric!"

# Close the database connection
conn.close()




