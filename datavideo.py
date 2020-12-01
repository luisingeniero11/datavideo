import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import utils
from sklearn import  metrics, svm
from sklearn.metrics import confusion_matrix
import pickle as joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import PowerTransformer

dataset = 'datavideo.csv'

#nombres = ['src_ip','dst_ip','f_pktTotalCount','b_pktTotalCount','f_octetTotalCount','b_octetTotalCount','f_avg_piat','b_avg_piat','f_avg_ps','b_avg_ps','src_port','dst_port','proto','f_std_dev_piat','b_std_dev_piat','f_std_dev_ps','b_std_dev_ps','Class']
#nombres = ['src_ip','dst_ip','bi_pktTotalCount','bi_octetTotalCount','bi_avg_piat','bi_avg_ps','src_port','dst_port','proto','bi_std_dev_piat','bi_std_dev_ps','bi_flowDuration','Class']
#data = pd.read_csv(dataset, names=nombres)
df_data = pd.read_csv(dataset)

df_data.drop(['src_ip','dst_ip'], axis = 1, inplace=True)
df_data = df_data.replace(np.nan, 0)

#features = df_data[['src_port','f_pktTotalCount','b_pktTotalCount','f_octetTotalCount','b_octetTotalCount','f_avg_piat','b_avg_piat','f_avg_ps','b_avg_ps','f_std_dev_piat','b_std_dev_piat','f_std_dev_ps','b_std_dev_ps']]
features = df_data[['src_port','src2dst_packets','dst2src_packets','src2dst_ip_bytes','dst2src_ip_bytes','src2dst_mean_piat_ms','dst2src_mean_piat_ms','src2dst_mean_ip_ps','dst2src_mean_ip_ps','src2dst_stdev_piat_ms','dst2src_stdev_piat_ms','src2dst_stdev_ip_ps','dst2src_stdev_ip_ps']] 

pt = PowerTransformer(method='yeo-johnson', standardize=True)

skl_boxcox = pt.fit(features)
clac_lambdas = skl_boxcox.lambdas_
#Fit the data to the powertransformer
skl_boxcox = pt.transform(features)
#Transform the data 
df_features = pd.DataFrame(data=skl_boxcox, columns = ['src_port','src2dst_packets','dst2src_packets','src2dst_ip_bytes','dst2src_ip_bytes','src2dst_mean_piat_ms','dst2src_mean_piat_ms','src2dst_mean_ip_ps','dst2src_mean_ip_ps','src2dst_stdev_piat_ms','dst2src_stdev_piat_ms','src2dst_stdev_ip_ps','dst2src_stdev_ip_ps'])



df_data2 = pd.concat([df_data['Class'],df_features], axis = 1)

cols2 = df_data2.columns.tolist()
for i in range(0, 13): 
    cols2 = cols2[-1:] + cols2[:-1]
    ++i   
df_data3 = df_data2[cols2]

array_clas3 = df_data3.values
X_cla3 = array_clas3[:,0:13]
Y_cla3 = array_clas3[:,13]

models = []

# evaluate each model in turn
models.append(('LoR', LogisticRegression(solver= 'lbfgs', max_iter=1000)))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('k-NN',KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree')))
models.append(('CART',DecisionTreeClassifier(criterion='entropy')))
models.append(('NB',GaussianNB()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_cla3, Y_cla3, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('name', cv_results.mean()*100, cv_results.std()*100)


#joblib.dump(dt, open('video-model3.pkl','wb')) #para crear el modelo de ML











