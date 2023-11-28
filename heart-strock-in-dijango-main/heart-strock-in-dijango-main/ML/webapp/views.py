import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier
from django.contrib import admin
from django.shortcuts import render
from django.contrib import messages


def index(request):
    return render(request,'index.html')


def about(request):
    return render(request,'about.html')

# def split():
#     df = pd.read_csv(r'M:\python&pycharm\archive\Heart Disease Dataset.csv')

#     x = df.iloc[:,:-1]
#     y = df.iloc[:,-1]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#     return 

def training(request):
    global dta,rfa,svma,knna,cba,df
    df = pd.read_csv(r'F:\python&pycharm\archive\Heart Disease Dataset.csv')

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    if request.method == 'POST':
        print("dfsdfdfsd")
        models = int(request.POST['algo'])
        if models == 1:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            dtp = dt.predict(x_test)
            dta = accuracy_score(y_test, dtp)
            dta = dta*100
            msg = 'Accuracy for Decision Tree is : ' + str(dta)
            return render(request,'training.html', {'msg':msg})
        elif models == 2:
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            rfp = rf.predict(x_test)
            rfa = accuracy_score(y_test, rfp)
            rfa = rfa*100
            msg = 'Accuracy for Random Forest is : ' + str(rfa)
            print(msg)
            return render(request,'training.html', {'msg':msg})

        elif models == 3:
            svm = SVC()
            svm.fit(x_train, y_train)
            svmp = svm.predict(x_test)
            svma = accuracy_score(y_test, svmp)
            svma = svma*100
            msg = 'Accuracy for SVM is : ' +str(svma)
            return render(request,'training.html', {'msg':msg})
        elif models == 4:
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            knnp = knn.predict(x_test)
            knna = accuracy_score(y_test, knnp)
            knna = knna*100
            msg = 'Accuracy for KNN is : ' + str(knna)

            return render(request,"training.html",{'msg':msg})
        # elif models == 5:
        #     cb = CatBoostClassifier()
        #     cb.fit(x_train, y_train)
        #     cbp = cb.predict(x_test)
        #     cba = accuracy_score(y_test, cbp)
        #     cba = cba*100
        #     msg = 'Accuracy for CatBoost is : ' + str(cba)

        #     return render(request,"training.html",{'msg':msg})

    return render(request,"training.html")

def prediction(request):
    
    global x_train, y_train,df
    if request.method == 'POST':
        print('aaaaaa')
        print('222222222')

        f1 = request.POST['f1']
        f2 = request.POST['f2']
        f3 = request.POST['f3']
        f4 = request.POST['f4']
        f5 = request.POST['f5']
        f6 = request.POST['f6']
        f7 = request.POST['f7']
        f8 = request.POST['f8']
        f9 = request.POST['f9']
        f10 = request.POST['f10']
        f11 = request.POST['f11']
        f12 = request.POST['f12']
        f13 = request.POST['f13']

        m = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13]
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        result = rf.predict([m])
        if result == 0:
            print("==================")            
            msg = 'The patient no heart stroke'
            messages.success(request, 'The patient no heart stroke')
        else:
            print("-------------------")
            msg = 'The patient has heart stroke'
            messages.success(request, 'The patient has heart stroke')


        return render(request,"prediction.html",{'msg':msg})
    return render(request,"prediction.html")

def chart(request):
    global dta,rfa,svma,knna
    i = [dta,rfa,svma,knna]
    return render(request,'chart.html',{'i':i})

