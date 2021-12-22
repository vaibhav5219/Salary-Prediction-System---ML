# Salary prediction system - for companies of HR department

# 1) Huge Data set
# 2) Create ML Model
# 3) Measure Performance of our ML model
# 4) Ready to use


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  #train_test_split is a function 
from sklearn.linear_model import LinearRegression # LinearRegresiio is class in sklearn library
from sklearn.metrics import r2_score

def welcome():
    print("Welcome to Salary Prediction System ")
    print("Press Enter key to Proceed... ")
    input()

def checkcsv(): # it will return collection of csv files
    csv_files = []
    cur_dir = os.getcwd() # getcwd() will return location/address of current dir
    content_list = os.listdir(cur_dir) # listdir returns list of all files of current dir
    for cf in content_list:
        if cf.split('.')[-1] == 'csv': #for finding extension of file
            csv_files.append(cf)   # appending the csv files only
    if len(csv_files)==0:       # ydi length = 0 hai to programme end ho jayega
        return "No csv file found"
    else:
        return csv_files

def display_and_select_csv(csv_files):
    i=1
    for file_name in csv_files:
        print(i,".> ",file_name)
        i+=1
    print()
    return csv_files[int(input("Select file to create ML model :- "))-1]

def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best Fit')
    plt.scatter(X_test,Y_test,color='green',label = 'test data')
    plt.scatter(X_test,Y_pred,color='black',label = 'Pred test data')
    plt.title("salary vs Experience")
    plt.xlabel("Years of experience")
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
    
def main():
    welcome()
    try:
        csv_files = checkcsv() # List of csv files
        if csv_files == "No csv file found":
            raise FileNotFoundError("No csv file found in the directory")
        csv_file = display_and_select_csv(csv_files)
        print(csv_file," is selected")
        print('Reading csv file')
        
        print('Creating data-set')
        dataset = pd.read_csv(csv_file)
        print('Data-set created')
        # obtaining  X and Y
        X = dataset.iloc[:,:-1].values # -1 index(salary col.) ko chhor kr sbhi col. ka data aa jaye
        Y = dataset.iloc[:,-1].values # salary column ka data bn jayega

        #splitting X and Y into testing
        s = float(input(" Enter test data size (between 0 and 1) :- "))
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=s) # it will split training and testing data
        print("Model creation in progression ")
    #important part of the project
        regressionObject = LinearRegression()
        regressionObject.fit(X_train, Y_train) # it will create BEST FIX regression line on trainning data
        print("Model is created ")
        print("Press Enter Key to predict test data in trained model ... ")
        input()

        Y_pred = regressionObject.predict(X_test) #predicted data of existing test data of the csv file
        i=0
        print(X_test,'  ....',Y_test,'  ....',Y_pred)
        while i<len(X_test):
            print(X_test[i], '...',Y_test[i],'....',Y_pred[i]) #test data, actual data predicted and
            i+=1
        print("Press Enter key to see above result in graphical format :- ")
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
        r2 = r2_score(Y_test,Y_pred)
        input("Our model is %2.2f%% accurate" %(r2*100))
        print()
        #USING ML MODEL
        print("Now you can predict salary of an employee using our model ")
        print("\nEnter experience in years of the candidates, separated by comma :- \n ")

        exp = [float(e) for e in input().split(',')]
        ex = []
        for x in exp:
            ex.append([x])
        experiences = np.array(ex)
        salaries = regressionObject.predict(experiences) # salary will be predicted

        plt.scatter(experiences,salaries,color='black')
        plt.xlabel('Years of Experiences')
        plt.ylabel('Salaries')
        plt.show()  # showing user's salary

        d = pd.DataFrame({'Experiences':exp, 'Salaries':salaries})
        print(d) #it will print in tabular form
        print("\n End of programme")    
        
    except FileNotFoundError:
        print("No csv file in the directory")
        print("Press Enter key to exit")
        input()
        exit()

if __name__ == "__main__":
    main()
    input()
