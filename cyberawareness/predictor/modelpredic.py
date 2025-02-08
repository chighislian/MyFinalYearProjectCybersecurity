import numpy as np
import pandas as pd
import joblib


from sklearn.tree import DecisionTreeClassifier



import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the dataset from excel using pandas
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")
# print(data.head())

# check for missing values / cleaning the dataSet
missing_values = data.isnull()  # This method returns  True where data is missing and False where data is present.
print(missing_values)
missing_values_per_column = data.isnull().sum()  # This returns the number of missing values per column
print(missing_values_per_column)
print(data.dtypes)  # Use the dtypes attribute to check the data types of each column.
data.info()  # Use this method to get a more detailed summary including data types and non-null counts for each column
# No missing value found

# mapping each response to numeric score To get the awareness Total

mapping_CybersecurityFamiliarity = {
    'Very familiar': 5,
    'Somewhat familiar': 3,
    'Not familiar': 0
}
data['CybersecurityFamiliarity'] = data['CybersecurityFamiliarity'].map(mapping_CybersecurityFamiliarity)

# mapping for understanding of phishing
mapping_phishing = {
    'I have never heard of it': 0,
    "I have heard of it but don't know what it is": 5,
    'I know what it is and can i can identify phishing attempts': 10,
    'I know what it is but find it hard to identify phishing': 15
}
data['PhishingUnderstanding'] = data['PhishingUnderstanding'].map(mapping_phishing)

# mapping for FirewallKnowledge

mapping_FirewallKnowledge = {
    'Yes, I know in detail': 10,
    'I have a basic understanding': 7,
    "I have heard of it but don't know how it works": 3,
    "No, I don't know": 0
}
data['FirewallKnowledge'] = data['FirewallKnowledge'].map(mapping_FirewallKnowledge)

# mapping for PasswordUpdateFrequency
mapping_PasswordUpdateFrequency = {
    'Monthly': 10,
    'Every few months': 7,
    'Once a year': 3,
    'Never': 0
}
data['PasswordUpdateFrequency'] = data['PasswordUpdateFrequency'].map(mapping_PasswordUpdateFrequency)

# mapping  SamePasswordUsage
mapping_SamePasswordUsage = {
    'Yes, for most accounts': 10,
    'Yes, but only for a few accounts': 5,
    'No, I use different passwords for all accounts': 0
}
data['SamePasswordUsage'] = data['SamePasswordUsage'].map(mapping_SamePasswordUsage)

# mapping TwoFactorAuthUsage
mapping_TwoFactorAuthUsage = {
    'Yes, for all accounts': 10,
    'Yes, but only for some accounts': 5,
    'No': 0
}
data['TwoFactorAuthUsage'] = data['TwoFactorAuthUsage'].map(mapping_TwoFactorAuthUsage)

# mapping  UnknownEmailResponse
mapping_UnknownEmailResponse = {
    'Open the attachment to see what it is': 0,
    'Delete the email immediately': 5,
    'Scan the attachment with antivirus software before opening': 7,
    'Report it as spam/phishing': 10
}
data['UnknownEmailResponse'] = data['UnknownEmailResponse'].map(mapping_UnknownEmailResponse)

# CyberIncidentExperience
mapping_CyberIncidentExperience = {
    'Yes': 0,
    'No': 1
}
data['CyberIncidentExperience'] = data['CyberIncidentExperience'].map(mapping_CyberIncidentExperience)

# IncidentResponse
mapping_IncidentResponse = {
    'Changed passwords': 1,
    'Reported it to the relevant  authorities': 2,
    'Did nothing': 0,
    'Contacted the service provider': 3
}
data['IncidentResponse'] = data['IncidentResponse'].map(mapping_IncidentResponse)

# mapping PhishingEmailResponse
mapping_PhishingEmailResponse = {
    'Clink the link and update your information': 0,
    'Ignore the email': 5,
    'Mark the email as spam': 7,
    'Contact the bank directly to verify the email': 10
}
data['PhishingEmailResponse'] = data['PhishingEmailResponse'].map(mapping_PhishingEmailResponse)

# mapping PopUpAlertAction
mapping_PopUpAlertAction = {
    'Run your antivirus software to check for issues': 10,
    'Close the pop-up and continue browsing': 7,
    'Restart your computer': 5,
    'Download and install the software immediately': 0
}
data['PopUpAlertAction'] = data['PopUpAlertAction'].map(mapping_PopUpAlertAction)

# mapping CybersecurityImportance
mapping_CybersecurityImportance = {
    'Extremely important': 5,
    'Very important': 3,
    'Somewhat important': 1,
    'Not important': 0
}
data['CybersecurityImportance'] = data['CybersecurityImportance'].map(mapping_CybersecurityImportance)

# mapping for DataProtectionConfidence
mapping_DataProtectionConfidence = {
    'Very confident': 5,
    'Somewhat confident': 3,
    'Not very confident': 1,
    'Not confident at all': 0
}
data['DataProtectionConfidence'] = data['DataProtectionConfidence'].map(mapping_DataProtectionConfidence)

# mapping CyberTrainingInterest
mapping_CyberTrainingInterest = {
    'Yes': 5,
    'Maybe': 3,
    'No': 0
}
data['CyberTrainingInterest'] = data['CyberTrainingInterest'].map(mapping_CyberTrainingInterest)

# Trying to Sum up the mapping score for each student

data['TotalScore'] = data[['CybersecurityFamiliarity', 'PhishingUnderstanding', 'FirewallKnowledge',
                           'SamePasswordUsage', 'TwoFactorAuthUsage', 'UnknownEmailResponse',
                           'CyberIncidentExperience', 'IncidentResponse', 'PhishingEmailResponse',
                           'PopUpAlertAction', 'CybersecurityImportance', 'DataProtectionConfidence',
                           'CyberTrainingInterest', 'PasswordUpdateFrequency']].sum(axis=1)

# Maximum possible score
max_score = 109

# Calculate the CyberAwarenessPercentage
data['CyberAwarenessPercentage'] = (data['TotalScore'] / max_score) * 100

# Print TotalScore and CyberAwarenessPercentage
print(data[['TotalScore', 'CyberAwarenessPercentage']])

# Creating the target variable based on the awareness percentage
bins = [0, 40, 70, 100]
labels = ['Low Awareness',  'Medium Awareness', 'High Awareness']
data['AwarenessLevel'] = pd.cut(data['CyberAwarenessPercentage'], bins=bins, labels=labels)
print(data['AwarenessLevel'])

# this step we label  the data into feature X and Target Y
X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']

# This step we split the data into both the Training (80%) and Testing (20%)
X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# This step we choose the model that is suitable for project based on the dataset we're working on
# We choose Random Forest Classifier model
# rf stand for random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# This step we train the model with the train data X_train/Y_train
rf_model.fit(X_train,Y_train)

# This step we test the performance of the model using X_test
Y_predict = rf_model.predict(X_test)
#joblib.dump(rf_model, 'ml_model.pkl')
joblib.dump(rf_model, 'cyber_awareness_model.pkl')
