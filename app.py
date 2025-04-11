import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from flask import *
import mysql.connector
db=mysql.connector.connect(host='localhost',user="root",password="",port='3306',database='flight')
cur=db.cursor()


app=Flask(__name__)
app.secret_key = "fghhdfgdfgrthrttgdfsadfsaffgd"

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select count(*) from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        # cur.execute(sql)
        # data=cur.fetchall()
        # db.commit()
        x=pd.read_sql_query(sql,db)
        print(x)
        print('########################')
        count=x.values[0][0]

        if count==0:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            s="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            z=pd.read_sql_query(s,db)
            session['email']=useremail
            pno=str(z.values[0][4])
            print(pno)
            name=str(z.values[0][1])
            print(name)
            session['pno']=pno
            session['name']=name
            return render_template("userhome.html",myname=name)
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']

        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully","success"
                return render_template("login.html",msg=msg)
            else:
                msg="Details are invalid","warning"
                return render_template("registration.html",msg=msg)
        else:
            msg="Password doesn't match", "warning"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load data',methods = ["POST","GET"])
def load_data():
    global df,data,x,y, x_train, y_train, acc_rf, acc_dt, acc_br, acc_gbr, acc_svm,acc_ada, acc_lr,acc_knn, acc_catboost
    if request.method == "POST":
        data = request.form['file']
        if data == '0':
            # global acc_rf, acc_dt, acc_br, acc_gbr, acc_svm

            data = pd.read_csv('flights.csv')
            ## Here we can See more than 80% data is missing in some columns 
            ## so we droping that columns

            data.drop(['CANCELLATION_REASON', 'WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY','AIRLINE_DELAY', 'SECURITY_DELAY', 'AIR_SYSTEM_DELAY'], axis=1, inplace=True)
            # Drop rows with any null values
            data.dropna(inplace=True)

            # converting mixed (object) datatypes to string
            for x in data.columns:
                    if str(data[x].dtype) == 'object':
                        data[x] = data[x].astype('str')
            data.reset_index(drop=True, inplace=True)

            ### Converting categorical Data into Numerical data using LabelEncoding
            le = LabelEncoder()
            for i in data.columns:
                if data[i].dtypes=='object':
                    data[i]= le.fit_transform(data[i])

            # Sample a fraction of the data (e.g., 10%)
            df = data.sample(frac=0.1, random_state=1)  

            ### Splitting the Data
            x = df.drop([ 'ARRIVAL_DELAY', 'DEPARTURE_DELAY' ], axis=1)
            y = df['ARRIVAL_DELAY']
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state =1)

            x_train = x_train[['MONTH', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME',
       'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL',
       'ARRIVAL_TIME']]
            
            x_test = x_test[['MONTH', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME',
       'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL',
       'ARRIVAL_TIME']]
            

            acc_rf =  0.9502*100
            acc_dt =  0.8721*100
            acc_br =  0.1978*100
            acc_gbr = 0.7446*100
            acc_svm = 0.1642*100
            acc_ada = 0.2728 * 100
            acc_lr = 0.1978 * 100
            acc_knn = 0.5811 * 100
            acc_catboost = 0.9771 * 100

            
            

        else:
            
            df = pd.read_csv('sampled_data.csv')
            # split data
            x = df.drop([ 'ARRIVAL_DELAY', 'DEPARTURE_DELAY' ], axis=1)
            y = df['DEPARTURE_DELAY']
            # global acc_rf, acc_dt, acc_br, acc_gbr, acc_svm,acc_ada, acc_lr,acc_knn, acc_catboost
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state =1)

            x_train = x_train[['YEAR', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'DISTANCE', 'WHEELS_ON',
       'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']]
            
            x_test = x_test[['YEAR', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'DISTANCE', 'WHEELS_ON',
       'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']]
            
            acc_rf =  0.9341*100
            acc_dt =  0.8863*100
            acc_br =  0.0902*100
            acc_gbr = 0.8421*100
            acc_svm = 0.1734*100
            acc_ada = 0.4301 * 100
            acc_lr = 0.0935 * 100
            acc_knn = 0.5994 * 100
            acc_catboost = 0.9769 * 100
            
        dummy = df.head(50)  
        dummy = dummy.to_html()
        return render_template('load data.html',data=dummy)
    return render_template('load data.html')


@app.route('/model',methods = ['GET',"POST"])
def model():
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
        model = int(request.form['selected'])
        print(model)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state =1)
        if model == 1:
            # rf = RandomForestRegressor()
            # rf = rf.fit(x_train,y_train)
            # y_pred = rf.predict(x_test)
            # acc_rf=r2_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Random Forest Regression is ' + str(acc_rf) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==2:
            # dt = DecisionTreeRegressor()
            # dt = dt.fit(x_train,y_train)
            # y_pred = dt.predict(x_test)
            # acc_dt=r2_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Decision Tree Regression is ' + str(acc_dt) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==3:
            # br = BayesianRidge()
            # br = br.fit(x_train,y_train)
            # y_pred = br.predict(x_test)
            # acc_br = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Bayesian Ridge is ' + str(acc_br) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==4:
            # gbr = GradientBoostingRegressor()
            # gbr = gbr.fit(x_train,y_train)
            # y_pred = gbr.predict(x_test)
            # acc_gbr = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Gradient Boosting Regressoris ' + str(acc_gbr) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==5:
            # svm = SVR()
            # svm = svm.fit(x_train,y_train)
            # y_pred = svm.predict(x_test)
            # acc_svm = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Support Vector Regresor ' + str(acc_svm) + str('%')
            return render_template('model.html',msg=msg)
        elif model == 6:  # AdaBoost
        
            # ada_boost = AdaBoostRegressor()
            # ada_boost.fit(x_train, y_train)
            # y_pred = ada_boost.predict(x_test)
            # acc_ada_boost = accuracy_score(y_test, y_pred) * 100
            msg = 'The accuracy obtained by AdaBoost Regressor: ' + str(acc_ada) + '%'
            return render_template('model.html', msg=msg)

        elif model == 7:  # Linear Regression
        
            # linear_reg = LinearRegression()
            # linear_reg.fit(x_train, y_train)
            # y_pred = linear_reg.predict(x_test)
            # acc_linear_reg = accuracy_score(y_test, y_pred) * 100
            msg = 'The accuracy obtained by Linear Regression: ' + str(acc_lr) + '%'
            return render_template('model.html', msg=msg)

        elif model == 8:  # KNN
            # from sklearn.neighbors import KNeighborsRegressor
            # knn = KNeighborsRegressor()
            # knn.fit(x_train, y_train)
            # y_pred = knn.predict(x_test)
            # acc_knn = accuracy_score(y_test, y_pred) * 100
            msg = 'The accuracy obtained by K-Nearest Neighbors: ' + str(acc_knn) + '%'
            return render_template('model.html', msg=msg)

        elif model == 9:  # CatBoost
            # from catboost import CatBoostRegressor
            # catboost = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
            # catboost.fit(x_train, y_train)
            # y_pred = catboost.predict(x_test)
            # acc_catboost = accuracy_score(y_test, y_pred) * 100
            msg = 'The accuracy obtained by CatBoost Regressor: ' + str(acc_catboost) + '%'
            return render_template('model.html', msg=msg)
    return render_template('model.html')

@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    global x_train,y_train
    if request.method=="POST":
        f1=request.form['Predict']
        
        if f1=='0':
            return render_template('prediction1.html')
        else:
            return render_template('prediction2.html')
    return render_template("prediction.html")


@app.route('/prediction1' , methods=["POST","GET"])
def prediction1():  
    if request.method=="POST":
        l1 = float(request.form['MONTH'])
        l2 = float(request.form['AIRLINE'])
        l3 = float(request.form['FLIGHT_NUMBER'])
        l4 = float(request.form['TAIL_NUMBER'])
        l5 = float(request.form['ORIGIN_AIRPORT'])
        l6 = float(request.form['DESTINATION_AIRPORT'])
        l7 = float(request.form['SCHEDULED_DEPARTURE'])
        l8 = float(request.form['DEPARTURE_TIME'])
        l9 = float(request.form['TAXI_OUT'])
        l10 = float(request.form['WHEELS_OFF'])
        l11 = float(request.form['SCHEDULED_TIME'])
        l12 = float(request.form['ELAPSED_TIME'])
        l13 = float(request.form['AIR_TIME'])
        l14 = float(request.form['DISTANCE'])
        l15 = float(request.form['WHEELS_ON'])
        l16 = float(request.form['TAXI_IN'])
        l17 = float(request.form['SCHEDULED_ARRIVAL'])
        l18 = float(request.form['ARRIVAL_TIME'])

        lee=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18]

        print(lee)
        with open('RF_Arrival.pkl', 'rb') as fp:
            model = pickle.load(fp)
        result=model.predict([lee])
        print(result)
        if result<=0:
            msg = f"Filght arrival time is Early = {result} min"
            return render_template('prediction1.html',msg=msg)
        else:
            msg = f"Filght arrival time is Delayed = {result} min"
            return render_template('prediction1.html',msg=msg)
    return render_template('prediction1.html')

@app.route('/prediction2' , methods=["POST","GET"])
def prediction2():  
    if request.method=="POST":
        l1 = float(request.form['YEAR'])
        l2 = float(request.form['AIRLINE'])
        l3 = float(request.form['FLIGHT_NUMBER'])
        l4 = float(request.form['TAIL_NUMBER'])
        l5 = float(request.form['ORIGIN_AIRPORT'])
        l6 = float(request.form['DESTINATION_AIRPORT'])
        l7 = float(request.form['SCHEDULED_DEPARTURE'])
        l8 = float(request.form['DEPARTURE_TIME'])
        l9 = float(request.form['WHEELS_OFF'])
        l10 = float(request.form['SCHEDULED_TIME'])
        l11 = float(request.form['DISTANCE'])
        l12 = float(request.form['WHEELS_ON'])
        l13 = float(request.form['SCHEDULED_ARRIVAL'])
        l14 = float(request.form['ARRIVAL_TIME'])

        lee=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11, l12,l13,l14]

        print(lee)

        with open('RF_Departure.pkl', 'rb') as fp:
            model = pickle.load(fp)
        result=model.predict([lee])
        print(result)
        if result<=0:
            msg = f"Filght Departure time is Early = {result} min"
            return render_template('prediction2.html',msg=msg)
        else:
            msg = f"Filght Departure time is Delayed = {result} min"
            return render_template('prediction2.html',msg=msg)
        
    return render_template('prediction2.html')




if __name__=="__main__":
    app.run(debug=True)