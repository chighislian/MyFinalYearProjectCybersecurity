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



# mapping each response to numeric score To get the awareness Total

# Mapping for CybersecurityFamiliarity
mapping_CybersecurityFamiliarity = {
    'Very familiar': 5,
    'Somewhat familiar': 3,
    'Not familiar': 0
}
data['CybersecurityFamiliarity'] = data['CybersecurityFamiliarity'].map(mapping_CybersecurityFamiliarity)

# Mapping for PhishingUnderstanding
mapping_phishing = {
    'I have never heard of it': 0,
    "I have heard of it but don't know what it is": 5,
    'I know what it is and I can identify phishing attempts': 15,
    'I know what it is but find it hard to identify phishing': 10
}
data['PhishingUnderstanding'] = data['PhishingUnderstanding'].map(mapping_phishing)

# Mapping for FirewallKnowledge
mapping_FirewallKnowledge = {
    'Yes, I know in detail': 10,
    'I have a basic understanding': 7,
    "I have heard of it but don't know how it works": 3,
    "No, I don't know": 0
}
data['FirewallKnowledge'] = data['FirewallKnowledge'].map(mapping_FirewallKnowledge)

# Mapping for PasswordUpdateFrequency
mapping_PasswordUpdateFrequency = {
    'Monthly': 10,
    'Every few months': 7,
    'Once a year': 3,
    'Never': 0
}
data['PasswordUpdateFrequency'] = data['PasswordUpdateFrequency'].map(mapping_PasswordUpdateFrequency)

# Mapping for SamePasswordUsage
mapping_SamePasswordUsage = {
    'Yes, for most accounts': 0,
    'Yes, but only for a few accounts': 5,
    'No, I use different passwords for all accounts': 10
}
data['SamePasswordUsage'] = data['SamePasswordUsage'].map(mapping_SamePasswordUsage)

# Mapping for TwoFactorAuthUsage
mapping_TwoFactorAuthUsage = {
    'Yes, for all accounts': 10,
    'Yes, but only for some accounts': 5,
    'No': 0
}
data['TwoFactorAuthUsage'] = data['TwoFactorAuthUsage'].map(mapping_TwoFactorAuthUsage)

# Mapping for UnknownEmailResponse
mapping_UnknownEmailResponse = {
    'Open the attachment to see what it is': 0,
    'Delete the email immediately': 5,
    'Scan the attachment with antivirus software before opening': 7,
    'Report it as spam/phishing': 10
}
data['UnknownEmailResponse'] = data['UnknownEmailResponse'].map(mapping_UnknownEmailResponse)

# Mapping for CyberIncidentExperience
mapping_CyberIncidentExperience = {
    'Yes': 1,  # Changed from 0 to 1 (since experience matters)
    'No': 0
}
data['CyberIncidentExperience'] = data['CyberIncidentExperience'].map(mapping_CyberIncidentExperience)

# Mapping for IncidentResponse
mapping_IncidentResponse = {
    'Changed passwords': 1,
    'Reported it to the relevant authorities': 3,
    'Did nothing': 0,
    'Contacted the service provider': 2
}
data['IncidentResponse'] = data['IncidentResponse'].map(mapping_IncidentResponse)

# Mapping for PhishingEmailResponse
mapping_PhishingEmailResponse = {
    'Click the link and update your information': 0,  # Fixed typo
    'Ignore the email': 5,
    'Mark the email as spam': 7,
    'Contact the bank directly to verify the email': 10
}
data['PhishingEmailResponse'] = data['PhishingEmailResponse'].map(mapping_PhishingEmailResponse)

# Mapping for PopUpAlertAction
mapping_PopUpAlertAction = {
    'Run your antivirus software to check for issues': 10,
    'Close the pop-up and continue browsing': 7,
    'Restart your computer': 5,
    'Download and install the software immediately': 0
}
data['PopUpAlertAction'] = data['PopUpAlertAction'].map(mapping_PopUpAlertAction)

# Mapping for CybersecurityImportance
mapping_CybersecurityImportance = {
    'Extremely important': 5,
    'Very important': 3,
    'Somewhat important': 1,
    'Not important': 0
}
data['CybersecurityImportance'] = data['CybersecurityImportance'].map(mapping_CybersecurityImportance)

# Mapping for DataProtectionConfidence
mapping_DataProtectionConfidence = {
    'Very confident': 5,
    'Somewhat confident': 3,
    'Not very confident': 1,
    'Not confident at all': 0
}
data['DataProtectionConfidence'] = data['DataProtectionConfidence'].map(mapping_DataProtectionConfidence)

# Mapping for CyberTrainingInterest
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

# Save the model to a file call cyber_awareness_model.pkl
#joblib.dump(rf_model, 'awareness_model.pkl')
#joblib.dump(rf_model, 'ml_model.pkl')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Assuming X_train, X_val, y_train, y_val are your training and validation data
train_accuracies = []
val_accuracies = []

for epoch in range(1, 51):  # Simulating 50 epochs
    rf_model.fit(X_train, Y_train)  # Retrain model in each iteration
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_test)

    train_acc = accuracy_score(Y_train, train_pred)
    val_acc = accuracy_score(Y_test, val_pred)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Save training history
history = {'train_accuracy': train_accuracies, 'val_accuracy': val_accuracies}

import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
from sklearn.metrics import log_loss

train_losses = []
val_losses = []

for epoch in range(1, 51):  # Simulating 50 epochs
    rf_model.fit(X_train, Y_train)  # Retrain model
    train_prob = rf_model.predict_proba(X_train)  # Get probabilities
    val_prob = rf_model.predict_proba(X_test)

    train_loss = log_loss(Y_train, train_prob)
    val_loss = log_loss(Y_test, val_prob)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Save training loss data
history["train_loss"] = train_losses
history["val_loss"] = val_losses

import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)