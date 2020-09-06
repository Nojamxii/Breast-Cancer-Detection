'''
Use extracted features to train Random Forests and save the parameters
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc

import utils as utils
import pickle

class RFClassifier:
      def __init__(self,parameter_path):
            '''
            parameter_path:
              RF model parameters, pickle format
            '''
            self.parameter_path = parameter_path

      def train(self,train_feature_path):
            '''
            Train the model and save parameters
            '''
            df_train = pd.read_csv(train_feature_path)
            print(df_train,end='\n********************************\n\n')

            n_columns = len(df_train.columns)
            feature_column_names = df_train.columns[:n_columns - 1]
            label_column_name = df_train.columns[n_columns - 1]

            train_x = df_train[feature_column_names]
            train_y = df_train[label_column_name]
            clf = RandomForestClassifier(n_estimators=50, n_jobs=2, verbose=1)
            clf.fit(X=train_x, y=train_y)

            
            with open(self.parameter_path,'wb') as f:
              pickle.dump(clf,f)
            print('The Random Forest model parameter was save in %s' %self.parameter_path)



      def predict(self,df_test):
            '''
            Args:
              test_feature_df: feature dataframe for test
            Load the parameters and make prediction
            '''
            
            feature_column_names = df_test.columns[:]
            test_x = df_test[:]

            clf = None
            with open(self.parameter_path,'rb') as f:
              clf = pickle.load(f)

            test_y = clf.predict(test_x)
            test_y_proba = clf.predict_proba(test_x)

            feature_importance = {}
            for key,value in zip(feature_column_names,clf.feature_importances_):
                  feature_importance[key] = value

            return test_y,test_y_proba,feature_importance

          


# df_train = pd.read_csv(utils.HEATMAP_FEATURE_CSV_TRAIN)
# df_validation = pd.read_csv(utils.HEATMAP_FEATURE_CSV_TEST)

# print(df_train, end='\n**************************************\n\n')
# print(df_validation)

# n_columns = len(df_train.columns)

# feature_column_names = df_train.columns[:n_columns - 1]
# label_column_name = df_train.columns[n_columns - 1]
# print(feature_column_names)
# print(label_column_name)

# train_x = df_train[feature_column_names]
# train_y = df_train[label_column_name]
# validation_x = df_validation[feature_column_names]
# validation_y = df_validation[label_column_name]

# clf = RandomForestClassifier(n_estimators=50, n_jobs=2,verbose=1)
# clf.fit(train_x, train_y)



# #eval the importance
# importances = clf.feature_importances_
# for f in range(train_x.shape[1]):
#   print("%2d) %-*s %f" % (f + 1, 30, feature_column_names[f], importances[f]))


# f, ax = plt.subplots(figsize=(7, 5))
# ax.bar(range(len(clf.feature_importances_)),clf.feature_importances_)
# ax.set_title("Feature Importances")
# f.show()
# f.savefig('/home/albelt/importance2_all.jpg')



# predict_y_validation = clf.predict(validation_x)
# print(predict_y_validation)
# prob_predict_y_validation = clf.predict_proba(validation_x)
# print(prob_predict_y_validation)


# #Export validation predict to json file
# predict_y_validation_json = pd.DataFrame(data=predict_y_validation)
# prob_predict_y_validation_json = pd.DataFrame(data=prob_predict_y_validation)
# predict_y_validation_json.to_json(utils.PREDICT_VALIDATION)
# prob_predict_y_validation_json.to_json(utils.PROB_PREDICT_VALIDATION)
# print('The validation predictions was save in: {}'.format(utils.PREDICT_VALIDATION))
# print('The validation probabilities was save in: {}'.format(utils.PROB_PREDICT_VALIDATION))



# predictions_validation = prob_predict_y_validation[:, 1]
# fpr, tpr, _ = roc_curve(validation_y, predictions_validation)
# roc_auc = auc(fpr, tpr)

# plt.title('ROC Validation')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# print(pd.crosstab(validation_y, predict_y_validation, rownames=['Actual'], colnames=['Predicted']))

# predict_y_train = clf.predict(train_x)
# print(predict_y_train)
# prob_predict_y_train = clf.predict_proba(train_x)
# print(prob_predict_y_train)

# #Export train predict to json file
# predict_y_train_json = pd.DataFrame(data=predict_y_train)
# prob_predict_y_train_json = pd.DataFrame(data=prob_predict_y_train)
# predict_y_train_json.to_json(utils.PREDICT_TRAIN)
# prob_predict_y_train_json.to_json(utils.PROB_PREDICT_TRAIN)
# print('The train predictions was save in: {}'.format(utils.PREDICT_TRAIN))
# print('The train probabilities was save in: {}'.format(utils.PROB_PREDICT_TRAIN))


# predictions_train = prob_predict_y_train[:, 1]
# fpr, tpr, _ = roc_curve(train_y, predictions_train)
# roc_auc = auc(fpr, tpr)

# plt.title('ROC Train')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# print(pd.crosstab(train_y, predict_y_train, rownames=['Actual'], colnames=['Predicted']))

