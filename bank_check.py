# _*_ coding : utf-8 _*_
# Time : 2018/7/26
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt

data = pd.read_csv("bank_check.csv")
# print(data.head())
# count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

# 数据预处理
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data[['Amount']])
data.drop(['Time','Amount'],axis=1,inplace=True)
# print(data)
X = data.ix[:,data.columns!='Class']
y = data.ix[:,data.columns=='Class']

# 下采样
number_records_negative = len(data[data['Class']==1])
negative_indexis = np.array(data[data['Class']==1].index)

positive_indexis = np.array(data[data['Class']==0].index)

random_positive_indexis = np.random.choice(positive_indexis,number_records_negative,replace=False)

under_sample_indexis = np.concatenate([negative_indexis,random_positive_indexis])
under_sample_data = data.ix[under_sample_indexis,:]

X_underSample = under_sample_data.ix[:,under_sample_data.columns!='Class']
y_underSample = under_sample_data.ix[:,under_sample_data.columns=='Class']

# ==========================下采样结束==============================

#===========================过采样==================================
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts
feature_columns = data.ix[:,data.columns!='Class']
label_columns = data.ix[:,data.columns=="Class"]
features_train,features_test,label_train,label_test = tts(feature_columns,label_columns,test_size=0.2,random_state=None)
ovsersample = SMOTE(random_state=0)
os_feature,os_label = ovsersample.fit_sample(features_train,label_train)
#==========================过采样结束==================================

#===========================分割训练集与测试集=======================
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train_sample,X_test_sample,y_train_sample,y_test_sample = train_test_split(X_underSample,y_underSample,test_size=0.3,random_state=0)

#=========================分割结束====================================

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score,KFold
from sklearn.metrics import recall_score,confusion_matrix,classification_report

def printing_KFold_score(X_traindata,y_traindata):

    fold = KFold(len(y_traindata),5,shuffle=False)

    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_scores = []
        for iteration,index in enumerate(fold,start=1):

            lr = LogisticRegression(C=c_param,penalty='l1',random_state=None)
            # print(X_traindata.ix[index[0],:])
            lr.fit(X_traindata.iloc[index[0],:].values,y_traindata.iloc[index[0],:].values.ravel())

            y_pre_undersample = lr.predict(X_traindata.iloc[index[1],:].values)

            re_score = recall_score(y_traindata.iloc[index[1],:],y_pre_undersample)
            recall_scores.append(re_score)
            print('Iteration ', iteration, ': recall score = ', re_score)

        results_table.ix[j,'Mean recall score'] = np.mean(recall_scores)
        j = j+1
        print('')
        print('Mean recall score ', np.mean(recall_scores))
        print('')


    # best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    results_table = results_table.sort_values(by='Mean recall score',ascending=False)
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', results_table.ix[0,'C_parameter'])
    print('*********************************************************************************')

    return results_table.ix[0,'C_parameter']

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    best_c = printing_KFold_score(X_train_sample,y_train_sample)
    # print(X_train_sample.values)

    lr = LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_train_sample, y_train_sample.values.ravel())
    y_pred_undersample = lr.predict(X_test_sample.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_sample, y_pred_undersample)
    np.set_printoptions(precision=2)
    print('$$$$$$$$$$$$$$$$$$$$$$$$')
    print(cnf_matrix)
    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()