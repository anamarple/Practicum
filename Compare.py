# Compares accuracies using different classifying methods paired with different vectorizers
import pandas as pd
import pyodbc
import numpy as np
from Normalize import normalize_df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # CV
from sklearn.feature_extraction.text import TfidfVectorizer  # Tf-idf
from sklearn.svm import LinearSVC  # SVM
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.ensemble import RandomForestClassifier  # random forest
from Predict import predict_bucket2
from sklearn.metrics import *
import matplotlib.pyplot as plt

########################################################################################################################
# 1. Retrieve documents
# Note: If driver not found, copy this in terminal: C:\Users\amarple\Downloads\AccessDatabaseEngine_X64.exe /passive
# Read data in from 'File_Classifications_DB.accdb'
# conn_str = r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=V:\APLA\users\arm\File " \
#               r"Classifications DB.accdb "

conn_str = r"Driver={SQL Server};Server=dm-sqlexpress\sqlexpress;Database=DMFileClassification;" \
           "UID=DMFCUser;PWD=Ydo%A39&B0Sl;"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

SQL_Query = pd.read_sql_query("SELECT * FROM [dbo].[Data]", conn)
df = pd.DataFrame(SQL_Query, columns=['File', 'Path', 'Bucket', 'Bucket2'])

# df = pd.read_excel(r'C:\Users\amarple\Desktop\DSA Practicum\Data\Consolidated.xlsx')

########################################################################################################################
# 2. Pre-process/normalize df
df = normalize_df(df)

########################################################################################################################
# 3. Split df into test (2/3) and train (1/3) data
test_path, train_path, test_file, train_file, test_bucket, train_bucket, test_bucket2, train_bucket2, test_path_norm, \
train_path_norm, test_file_norm, train_file_norm = train_test_split(np.array(df['Path']), np.array(df['File']),
                                                                    np.array(df['Bucket']), np.array(df['Bucket2']),
                                                                    np.array(df['Normalized_Path']),
                                                                    np.array(df['Normalized_File']), test_size = 0.33,
                                                                    random_state = 42)

# Build train data frame
train_data = {'Path': train_path, 'Normalized_Path': train_path_norm, 'File': train_file, 'Normalized_File':
    train_file_norm, 'Target_Bucket': train_bucket, 'Target_Bucket2': train_bucket2}
df_train = pd.DataFrame(train_data)

# Split train df into 5 different data frames based on first level bucket:
df_train_gen = df_train[df_train['Target_Bucket'] == 'General']
df_train_comp = df_train[df_train['Target_Bucket'] == 'Completion']
df_train_geo = df_train[df_train['Target_Bucket'] == 'Geological']
df_train_prod = df_train[df_train['Target_Bucket'] == 'Production']
df_train_eco = df_train[df_train['Target_Bucket'] == 'Economics']

########################################################################################################################
# 4. Feature engineering
# Try and compare different methods: cv, tf-idf, word2vec (CBOW?)

# First level bucket extraction: based on path + file name
# CV - level 1:
cv = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
train_features_cv = cv.fit_transform(train_path_norm)
test_features_cv = cv.transform(test_path_norm)


# Returns cv for level 1 (so it can be used in main function)
def get_cv_l1():
    return cv


# Tf-idf - level 1:
tf = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
train_features_tf = tf.fit_transform(train_path_norm)
test_features_tf = tf.transform(test_path_norm)

# Second level bucket feature extraction: based on file name
# CV - level 2:
cv_comp = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
train_features_comp_cv = cv_comp.fit_transform(df_train_comp['Normalized_File'])
cv_geo = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
train_features_geo_cv = cv_geo.fit_transform(df_train_geo['Normalized_File'])
cv_prod = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
train_features_prod_cv = cv_prod.fit_transform(df_train_prod['Normalized_File'])
cv_gen = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
train_features_gen_cv = cv_gen.fit_transform(df_train_gen['Normalized_File'])


# Returns cv for level 2 (so it can be used in main function)
def get_cv_l2():
    return cv_comp, cv_geo, cv_prod, cv_gen


# Tf-idf - level 2:
tf_comp = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
train_features_comp_tf = tf_comp.fit_transform(df_train_comp['Normalized_File'])
tf_geo = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
train_features_geo_tf = tf_geo.fit_transform(df_train_geo['Normalized_File'])
tf_prod = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
train_features_prod_tf = tf_prod.fit_transform(df_train_prod['Normalized_File'])
tf_gen = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
train_features_gen_tf = tf_gen.fit_transform(df_train_gen['Normalized_File'])

########################################################################################################################
# 5. Build classifier
# Try and compare different methods: svm, logistic regression, random forest

# Train/ Prepare model for first level
# SVM - level 1:
# cv
svm_l1_cv = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l1_cv.fit(train_features_cv, train_bucket)
# tf-idf
svm_l1_tf = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l1_tf.fit(train_features_tf, train_bucket)

# Logistic Regression - level 1:
# cv
lr_1_cv = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_1_cv.fit(train_features_cv, train_bucket)
# tf-idf
lr_1_tf = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_1_tf.fit(train_features_tf, train_bucket)

# Random Forest - level 1:
# cv
rf_l1_cv = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l1_cv.fit(train_features_cv, train_bucket)
# tf-idf
rf_l1_tf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l1_tf.fit(train_features_tf, train_bucket)

# Train/ Prepare models for second level
# SVM - level 2:
# cv
svm_l2_comp_cv = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_comp_cv.fit(train_features_comp_cv, df_train_comp['Target_Bucket2'])
svm_l2_geo_cv = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_geo_cv.fit(train_features_geo_cv, df_train_geo['Target_Bucket2'])
svm_l2_prod_cv = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_prod_cv.fit(train_features_prod_cv, df_train_prod['Target_Bucket2'])
svm_l2_gen_cv = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_gen_cv.fit(train_features_gen_cv, df_train_gen['Target_Bucket2'])
# tf-idf
svm_l2_comp_tf = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_comp_tf.fit(train_features_comp_tf, df_train_comp['Target_Bucket2'])
svm_l2_geo_tf = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_geo_tf.fit(train_features_geo_tf, df_train_geo['Target_Bucket2'])
svm_l2_prod_tf = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_prod_tf.fit(train_features_prod_tf, df_train_prod['Target_Bucket2'])
svm_l2_gen_tf = LinearSVC(penalty = 'l2', C = 1, random_state = 42, max_iter = 10000)
svm_l2_gen_tf.fit(train_features_gen_tf, df_train_gen['Target_Bucket2'])

# Logistic Regression - level 2:
# cv
lr_l2_comp_cv = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_comp_cv.fit(train_features_comp_cv, df_train_comp['Target_Bucket2'])
lr_l2_geo_cv = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_geo_cv.fit(train_features_geo_cv, df_train_geo['Target_Bucket2'])
lr_l2_prod_cv = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_prod_cv.fit(train_features_prod_cv, df_train_prod['Target_Bucket2'])
lr_l2_gen_cv = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_gen_cv.fit(train_features_gen_cv, df_train_gen['Target_Bucket2'])
# tf-idf
lr_l2_comp_tf = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_comp_tf.fit(train_features_comp_tf, df_train_comp['Target_Bucket2'])
lr_l2_geo_tf = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_geo_tf.fit(train_features_geo_tf, df_train_geo['Target_Bucket2'])
lr_l2_prod_tf = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_prod_tf.fit(train_features_prod_tf, df_train_prod['Target_Bucket2'])
lr_l2_gen_tf = LogisticRegression(penalty = 'l2', max_iter = 100, C = 1, random_state = 42)
lr_l2_gen_tf.fit(train_features_gen_tf, df_train_gen['Target_Bucket2'])

# Random Forest - level 2:
# cv
rf_l2_comp_cv = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_comp_cv.fit(train_features_comp_cv, df_train_comp['Target_Bucket2'])
rf_l2_geo_cv = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_geo_cv.fit(train_features_geo_cv, df_train_geo['Target_Bucket2'])
rf_l2_prod_cv = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_prod_cv.fit(train_features_prod_cv, df_train_prod['Target_Bucket2'])
rf_l2_gen_cv = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_gen_cv.fit(train_features_gen_cv, df_train_gen['Target_Bucket2'])
# tf-idf
rf_l2_comp_tf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_comp_tf.fit(train_features_comp_tf, df_train_comp['Target_Bucket2'])
rf_l2_geo_tf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_geo_tf.fit(train_features_geo_tf, df_train_geo['Target_Bucket2'])
rf_l2_prod_tf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_prod_tf.fit(train_features_prod_tf, df_train_prod['Target_Bucket2'])
rf_l2_gen_tf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf_l2_gen_tf.fit(train_features_gen_tf, df_train_gen['Target_Bucket2'])


########################################################################################################################
# 6. Predict
# Feeds features from main function into best performing model (svm_l1_cv) and returns level 1 predictions
def predict_b1(features):
    x = svm_l1_cv.predict(features)
    return x


# Feeds df from main function into best performing level 2 models and returns level 2 predictions
def predict_b2(df_input, col, cv_comp_input, cv_geo_input, cv_prod_input, cv_gen_input):
    x = predict_bucket2(df_input, col, cv_comp_input, cv_geo_input, cv_prod_input, cv_gen_input,
                        svm_l2_comp_cv, svm_l2_geo_cv, svm_l2_prod_cv, svm_l2_gen_cv)
    return x


# Predict first bucket
# SVM:
# cv
y = svm_l1_cv.predict(test_features_cv)
# tf-idf
y2 = svm_l1_tf.predict(test_features_tf)

# Logistic Regression:
# cv
y3 = lr_1_cv.predict(test_features_cv)
# tf-idf
y4 = lr_1_tf.predict(test_features_tf)

# Random Forest:
# cv
y5 = rf_l1_cv.predict(test_features_cv)
# tf-idf
y6 = rf_l1_tf.predict(test_features_tf)

results = {'Path': test_path_norm, 'File': test_file_norm, 'Bucket': test_bucket, 'Predicted_svm_cv': y,
           'Predicted_svm_tf': y2, 'Predicted_lr_cv': y3, 'Predicted_lr_tf': y4, 'Predicted_rf_cv': y5,
           'Predicted_rf_tf': y6, 'Bucket2': test_bucket2}
results_df = pd.DataFrame(results)

# Predict second bucket
# SVM w/ cv:
bucket2_pred_svm_cv = predict_bucket2(results_df, 'Predicted_svm_cv', cv_comp, cv_geo, cv_prod, cv_gen, svm_l2_comp_cv,
                                      svm_l2_geo_cv, svm_l2_prod_cv, svm_l2_gen_cv)
results_df['Predicted_Bucket2_svm_cv'] = bucket2_pred_svm_cv
# SVM w/ tf-idf:
bucket2_pred_svm_tf = predict_bucket2(results_df, 'Predicted_svm_tf', tf_comp, tf_geo, tf_prod, tf_gen, svm_l2_comp_tf,
                                      svm_l2_geo_tf, svm_l2_prod_tf, svm_l2_gen_tf)
results_df['Predicted_Bucket2_svm_tf'] = bucket2_pred_svm_tf

# Logistic Regression w/ cv:
bucket2_pred_lr_cv = predict_bucket2(results_df, 'Predicted_lr_cv', cv_comp, cv_geo, cv_prod, cv_gen, lr_l2_comp_cv,
                                     lr_l2_geo_cv, lr_l2_prod_cv, lr_l2_gen_cv)
results_df['Predicted_Bucket2_lr_cv'] = bucket2_pred_lr_cv
# Logistic Regression w/ tf-idf:
bucket2_pred_lr_tf = predict_bucket2(results_df, 'Predicted_lr_tf', tf_comp, tf_geo, tf_prod, tf_gen, lr_l2_comp_tf,
                                     lr_l2_geo_tf, lr_l2_prod_tf, lr_l2_gen_tf)
results_df['Predicted_Bucket2_lr_tf'] = bucket2_pred_lr_tf

# Random Forest w/ cv:
bucket2_pred_rf_cv = predict_bucket2(results_df, 'Predicted_rf_cv', cv_comp, cv_geo, cv_prod, cv_gen, rf_l2_comp_cv,
                                     rf_l2_geo_cv, rf_l2_prod_cv, rf_l2_gen_cv)
results_df['Predicted_Bucket2_rf_cv'] = bucket2_pred_rf_cv
# Random Forest w/ tf-idf:
bucket2_pred_rf_tf = predict_bucket2(results_df, 'Predicted_rf_tf', tf_comp, tf_geo, tf_prod, tf_gen, rf_l2_comp_tf,
                                     rf_l2_geo_tf, rf_l2_prod_tf, rf_l2_gen_tf)
results_df['Predicted_Bucket2_rf_tf'] = bucket2_pred_rf_tf

# pd.set_option('display.max_rows', 1000)
# print(results_df)


########################################################################################################################
# 7. Evaluate models
# Find accuracy for each bucket level // create visualization of accuracy (+ runtime?)

# Plot Bar Chart
plt.title('Method vs. Accuracy')
plt.xlabel('Accuracy')

b1_acc = [accuracy_score(results_df['Bucket'], results_df['Predicted_svm_cv']),
          accuracy_score(results_df['Bucket'], results_df['Predicted_svm_tf']),
          accuracy_score(results_df['Bucket'], results_df['Predicted_lr_cv']),
          accuracy_score(results_df['Bucket'], results_df['Predicted_lr_tf']),
          accuracy_score(results_df['Bucket'], results_df['Predicted_rf_cv']),
          accuracy_score(results_df['Bucket'], results_df['Predicted_rf_tf'])]

b2_acc = [accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_svm_cv']),
          accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_svm_tf']),
          accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_lr_cv']),
          accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_lr_tf']),
          accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_rf_cv']),
          accuracy_score(results_df['Bucket2'], results_df['Predicted_Bucket2_rf_tf'])]

indices = np.arange(len(b1_acc))
methods = ['SVM w/ cv', 'SVM w/ tf-idf', 'Logistic Regression w/ cv', 'Logistic Regression w/ tf-idf',
           'Random Forest w/ cv', 'Random Forest w/ tf-idf']

plt.barh(indices, b1_acc, 0.2, label = 'Bucket 1 Accuracy', color = 'c')
plt.barh(indices + 0.2, b2_acc, 0.2, label = 'Bucket 2 Accuracy', color = 'blue')
plt.yticks(())
plt.legend(bbox_to_anchor = (-0.2, 1.15), loc = 'upper left')
plt.subplots_adjust(left = 0.25)

for i, c in zip(indices, methods):
    plt.text(-0.2, i, c)
for index, value in enumerate(b1_acc):
    plt.text(value, index, str(round(value, 4)))
for index, value in enumerate(b2_acc):
    plt.text(value, index + 0.2, str(round(value, 4)))

########################################################################################################################
# Error Analysis Extended:

''''
# Accuracy, Precision, Recall, F1 Score
# SVM w/ cv:
print('SVM with cv: \n')
class_rpt_svm_cv_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_svm_cv'])
print(class_rpt_svm_cv_b2)


# Confusion Matrix
labels = results_df.Bucket2.unique().tolist()
print(labels)

cm = confusion_matrix(results_df['Bucket2'], results_df['Predicted_Bucket2_svm_cv'],
                      labels = labels) #, normalize = 'true')
cm_display = ConfusionMatrixDisplay(cm).plot()


# SVM w/ tf-idf:
print('SVM with tf-idf: \n')
class_rpt_svm_tf_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_svm_tf'])
print(class_rpt_svm_tf_b2)

# Logistic Regression w/ cv:
print('Logistic Regression with cv: \n')
class_rpt_lr_cv_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_lr_cv'])
print(class_rpt_lr_cv_b2)

# Logistic Regression w/ tf-idf:
print('Logistic Regression with tf-idf: \n')
class_rpt_lr_tf_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_lr_tf'])
print(class_rpt_lr_tf_b2)

# Random Forest w/ cv:
print('Random Forest with cv: \n')
class_rpt_rf_cv_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_rf_cv'])
print(class_rpt_rf_cv_b2)

# Random Forest w/ tf-idf:
print('Random Forest with tf-idf: \n')
class_rpt_rf_tf_b2 = classification_report(results_df['Bucket2'], results_df['Predicted_Bucket2_rf_tf'])
print(class_rpt_rf_tf_b2)
'''''