# Importing necessary libraries
from re import T
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import copy,math
from sklearn.metrics import confusion_matrix

#--------------------------Information Gain Functions------------------------------

def B_value(q):
    return -( q*math.log(q,2) + (1-q)*math.log(1-q,2) )

def get_gain(Xi, Y):
    print(type(Xi))
    print(type(Y))
    Xi = Xi.reshape(Xi.shape[0],1)
    #Xi = Xi.T
    Xiy = np.concatenate([Xi,Y.T], axis=1)
    print(Xiy.shape)


def inf_gain(X, Y):
    p = Y.sum()
    n = X.shape[1] - p

    get_gain(X[0], Y)

    #print(X.shape)
    #print(Y.shape)
    

#--------------------------Functions for logistic Regression-----------------------
def tanh(x):
    t = (np.exp(x)-np.exp(-x)) /( np.exp(x)+np.exp(-x))
    return t

def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    ##print("w shape: "+str(w.shape))
    #print("x: shape"+ str(X.shape))
    A = tanh( np.dot(w.T, X) + b )
    #print("A" + str(A.shape))
    A = (A + 1.0)/2.0
    sq = np.square(A - Y)
    #print(sq.shape)
    
    #cost = np.sum(sq)/m
    cost = np.mean(sq, axis=1)
    #print(cost.shape)
    #print(cost)
    
    t = (Y-A)*(1-np.square(A))
    dw = -2 * np.dot(X,t.T)/m
    db = -2 * np.sum(t)/m
    
    db = np.squeeze(np.array(db)) 
    cost = np.squeeze(np.array(cost))
    #print(type(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, threshold):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    '''print("optimise: ")
    print(X.shape)
    print(Y.shape)'''

    for i in range(num_iterations):
        grads, cost = propagate(w,b,X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if cost < threshold:
            break
        
    params = {"w": copy.deepcopy(w),
              "b": copy.deepcopy(b)}
    
    return params


def accuracy(Y_test, Y_predicted_test):
    return float(100 - np.mean(np.abs(Y_predicted_test - Y_test)) * 100)


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = tanh( np.dot(w.T, X) + b )
    A = (A + 1.0)/2.0

    for i in range(A.shape[1]):        
        if A[0, i] > 0.5 :
             Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        
    return Y_prediction


def model(X_train, Y_train, num_iterations, learning_rate, threshold):
    #inf_gain(X_train, Y_train)

    w, b = initialize_with_zeros(X_train.shape[0])

    params = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, threshold)
    w = params["w"]
    b = params["b"]
    
    #Y_prediction_test = predict(w, b, X_test)
    #Y_prediction_train = predict(w, b, X_train)

    ##test_accuracy = accuracy(Y_test, Y_prediction_test)
    #print("Train Accuracy")
    #accuracy(Y_train, Y_prediction_train)
    
    d = {
         "w" : w, 
         "b" : b,
        }
    #print(d["w"])

    return d


#-----------------------1st Dataset Load functions---------------------------------
dataset1_file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
dataset2_file_name = 'adult.data'
dataset3_file_name = 'creditcard.csv'


def loadfirstfile(filename):
    data = pd.read_csv(filename)
    
    data = data.drop('customerID',axis=1)
    
    data = data.replace(r'^\s*$', np.nan, regex=True)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imp.fit(data[['TotalCharges']])
    data['TotalCharges'] = imputer.transform(data[['TotalCharges']])
    
    nominalcases = pd.get_dummies(data[['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']], drop_first=False).to_numpy()
    tfcases = pd.get_dummies(data[['Partner','Dependents','PhoneService', 'PaperlessBilling']], drop_first=True).to_numpy()
    scalar =  MinMaxScaler()
    minmaxscalarcases = scalar.fit_transform(data[['tenure','MonthlyCharges','TotalCharges']])
    
    concatenated = np.concatenate([nominalcases, tfcases, minmaxscalarcases], axis=1)

    target = pd.get_dummies(data[['Churn']], drop_first=True).to_numpy()
    
    return concatenated, target

def loadsecondfile(filename):
    data = pd.read_csv(filename)
    #print(data.shape)
    data = data.loc[(data[' State-gov']!=" ?") & (data[' Adm-clerical']!=" ?") & (data[' United-States']!=" ?") ]
    #print(data.shape)

    scalar =  MinMaxScaler()
    minmaxscalarcases = scalar.fit_transform(data[['39',' 77516',' 13',' 2174',' 0',' 40']])
    nominalcases = pd.get_dummies(data[[' State-gov',' Bachelors',' Never-married',' Adm-clerical', ' Not-in-family', ' White', ' Male', ' United-States']], drop_first=False).to_numpy()
    concatenated = np.concatenate([nominalcases, minmaxscalarcases], axis=1)

    target = pd.get_dummies(data[[' <=50K']], drop_first=True).to_numpy()
    #print(target.shape)
    
    return concatenated, target

def loadthirdfile(filename):
    data = pd.read_csv(filename)
    pos_cases = data.loc[data['Class']==1]
    neg_cases = data.loc[data['Class']==0]
    neg_cases = neg_cases[:5000]

    #print(neg_cases.shape)
    data = pd.concat([pos_cases,neg_cases])
    #print(data.shape)

    scalar =  MinMaxScaler()
    minmaxscalarcases = scalar.fit_transform(data[['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
    #print(minmaxscalarcases.shape)
    target = data[['Class']].to_numpy()
    return minmaxscalarcases, target


def loadonlinefile(filename):
    data = pd.read_csv(filename)
    scalar =  MinMaxScaler()
    minmaxscalarcases = scalar.fit_transform(data[['6','148','72','35','0','33.6', '0.627','50']])
    target = data[['1']].to_numpy()
    
    return minmaxscalarcases, target

def Adaboost(examples ,k):
    m = examples.shape[0]
    w = np.full( (1,m), 1/m, dtype=float)
    
    examples_x = examples[:examples.shape[0],:(examples.shape[1])-1]
    examples_y = examples[:examples.shape[0], (examples.shape[1])-1:(examples.shape[1])]
    examples_x = examples_x.transpose()
    examples_y = examples_y.transpose()

    z = []
    h = []
    for i in range(k):
        data_indices = np.random.choice(m, m, p=w[0])
        #print(data_indices.shape)
        data = np.take(examples, data_indices, 0)
        '''data = pd.DataFrame(examples).iloc[data_indices]
        data = data.to_numpy()
        print(data.shape)'''
        
        data_x = data[:data.shape[0],:(data.shape[1])-1]
        data_y = data[:data.shape[0], (data.shape[1])-1:(data.shape[1])]
        data_x = data_x.transpose()
        data_y = data_y.transpose()

        h_t = model(X_train=data_x, Y_train=data_y,
                    num_iterations=50, learning_rate=0.05,threshold=0.001)
        
        error = 0

        Y_prediction = predict(h_t["w"], h_t["b"], examples_x)
        #print(Y_prediction.shape)
        #print(data_y.shape)

        #print(accuracy(y_test, Y_prediction))
        for i in range(m):
            if Y_prediction[0][i] != examples_y[0][i]:
                error = error + w[0][i]
        
        z.append(math.log((1-error)/error , 2))
        h.append(h_t)

        if error>0.5:
            continue

        for i in range(m):
            if Y_prediction[0][i] == data_y[0][i]:
                w[0][i] = w[0][i] * (error/(1-error))
        w = w / w.sum()
    s = sum(z)
    for i in range(len(z)):
        z[i] = z[i]/s
    return h,z


def Adaboost_predict(h,z, X_test, Y_test):
    m = X_test.shape[1]
    Y_prediction = np.zeros((1, m))
        
    for i in range(len(h)):
        w = h[i]["w"]
        b = h[i]["b"]
        
        A = tanh( np.dot(w.T, X_test) + b )
        A = (A+1.0)/2
        Y_prediction = Y_prediction + np.multiply(A, z[i])        
    
    for i in range(Y_prediction.shape[1]):    
        if Y_prediction[0, i] > 0.5 :
             Y_prediction[0,i] = 1 
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction




#------------------------Dataset read commands---------------
#X, Y = loadfirstfile(dataset1_file_name)
#X, Y = loadsecondfile(dataset2_file_name)
#X, Y = loadthirdfile(dataset3_file_name)

X, Y = loadonlinefile("data.csv")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#------------------------Adaboost-------------------------
examples = np.concatenate([X_train, y_train], axis=1)
h,z = Adaboost(examples, 10)

y_adaboost_train_pred = Adaboost_predict(h=h,z=z,X_test=X_train.transpose(),Y_test = y_train.transpose())

print("Train accuracy : ")
print("{:.2f}".format(accuracy(Y_test=y_train,Y_predicted_test= y_adaboost_train_pred.transpose())))

y_adaboost_pred = Adaboost_predict(h=h,z=z,X_test=X_test.transpose(),Y_test = y_test.transpose())
print("Test Accuracy : ")
print("{:.2f}".format(accuracy(Y_test=y_test,Y_predicted_test= y_adaboost_pred.transpose())))



#--------------------Linear Regression----------------------------
'''d = model(X_train=X_train.transpose(), Y_train=y_train.transpose(),
                    num_iterations=5000, learning_rate=0.05, threshold= 0.05)

y_tr = predict(w=d['w'],b= d['b'], X = X_train.transpose()) 
print("Training dataset")
tn, fp, fn, tp = confusion_matrix(y_train.transpose()[0], y_tr[0]).ravel()

sensitivity = "{:.2f}".format(tp/(tp+fn) *100)
specificity = "{:.2f}".format(tn / (tn+fp) *100)
precision = "{:.2f}".format(tp/(tp+fp) *100)
false_discovery_rate = "{:.2f}".format(fp/(fp+tp) *100)
f1_score = "{:.2f}".format(2*tp/(2*tp + fp + fn) *100)
print("sensitivity : "+str(sensitivity))
print("specificity : "+ str(specificity))
print("precision :"+str(precision))
print("false discovery rate :"+str(false_discovery_rate))
print("f1 score : "+str(f1_score))

print(accuracy(Y_test = y_train,Y_predicted_test= y_tr.transpose()))

y_p = predict(w=d['w'],b= d['b'], X = X_test.transpose())
print("Test Dataset")
tn, fp, fn, tp = confusion_matrix(y_test.transpose()[0], y_p[0]).ravel()

sensitivity = "{:.2f}".format(tp/(tp+fn) *100)
specificity = "{:.2f}".format(tn / (tn+fp) *100)
precision = "{:.2f}".format(tp/(tp+fp) *100)
false_discovery_rate = "{:.2f}".format(fp/(fp+tp) *100)
f1_score = "{:.2f}".format(2*tp/(2*tp + fp + fn) *100)
print("sensitivity : "+str(sensitivity))
print("specificity : "+ str(specificity))
print("precision :"+str(precision))
print("false discovery rate :"+str(false_discovery_rate))
print("f1 score : "+str(f1_score))

print("{:.2f}".format(accuracy(Y_test = y_test,Y_predicted_test= y_p.transpose())))'''