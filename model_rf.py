

#Documentação do algoritmo http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

mypath = 'C:/Users/gtvol/Documents/campo/reduce/'

onlyfiles = ['29Oct_2015.csv', '10Nov_2015.csv', '22Nov_2015.csv', '04Dec_2015.csv', '21Jan_2016.csv', '14Feb_2016.csv', '09Mar_2016.csv', '21Mar_2016.csv', '08May_2016.csv', '13Jun_2016.csv', '07Jul_2016.csv', '31Jul_2016.csv']

for i in range(len(onlyfiles)):
    
    #junta caminhos + o nome dos arquivos
    path_file  = mypath+onlyfiles[i]
    df = pd.read_csv(path_file) # Le csv para dataframe

    y = df.pop('class') # remove a coluna de classes do y
    X = df #o resto das colunas fica no x    
    
    kf = KFold(n_splits=5) #KFOLD 5
    kf.get_n_splits(X)
    
    acc_kfold = []
    f1_kfold = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf=RandomForestClassifier(n_estimators=250,max_depth=25)    
        
        clf.fit(X_train, y_train) #Treina modelo    
        y_pred=clf.predict(X_test) #Predict do modelo      
        
        #Calcula as metricas
        f1_score = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for the fold no. {i} on the test set: {accuracy}")
        print(f"f1_score for the fold no. {i} on the test set: {f1_score}")
        
    print("Median Accuracy: {np.median(acc_kfold)} ")
    print("Median F1 Score: {np.median(f1_kfold)}")
    
    
    
    
