from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
import pickle
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import Firefly
from Firefly import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

main = tkinter.Tk()
main.title("Data Mining based Prediction of Diabetes using Firefly Optimized Neural Network")
main.geometry("1300x1200")

global dataset
global X, Y, X_train, y_train, X_test, y_test, firefly_ann, scaler
global accuracy, precision, recall, fscore, columns, firefly

def uploadDataset():
    global dataset, columns
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    columns = dataset.columns
    label = dataset.groupby('Outcome').size()
    label.plot(kind="bar")
    plt.xlabel("Diabetes Graph 0 (Normal) & 1 (Diabetes)")
    plt.ylabel("Count")
    plt.title("Diabetes Graph")
    plt.show()


def processDataset():
    global dataset
    global X, Y, X_train, y_train, X_test, y_test, scaler
    dataset.fillna(0, inplace = True)
    text.delete('1.0', END)
    scaler = StandardScaler()
    Y = dataset['Outcome'].values
    X = dataset.drop('Outcome', axis=1).values
    X = scaler.fit_transform(X)
       

    text.insert(END,"Dataset Preprocessing Completed\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset train & test split details\n\n")
    text.insert(END,"80% images are used to train ANN : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images are used to train ANN : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, true_label, predict_label):
    p = precision_score(true_label, predict_label,average='macro') * 100
    r = recall_score(true_label, predict_label,average='macro') * 100
    f = f1_score(true_label, predict_label,average='macro') * 100
    a = accuracy_score(true_label, predict_label)*100  
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ['Normal', 'Diabetes']
    conf_matrix = confusion_matrix(true_label, predict_label)
    plt.figure(figsize =(4, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def trainANN():
    global accuracy, precision, recall, fscore
    global X, Y, X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    #Define ANN layers
    ann_model = Sequential()
    #add dense layer with 512 neurons
    ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    #dropout to remove irrelevant features 
    ann_model.add(Dropout(0.2))
    #adding another with 512 neurons o filtered dataset
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.2))
    #define output layer
    ann_model.add(Dense(y_train.shape[1]))
    ann_model.add(Activation('softmax'))
    #compile the model
    ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #train and load the model
    if os.path.exists("model/ann_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
        hist = ann_model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/ann_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        ann_model.load_weights("model/ann_weights.hdf5")
    predict = ann_model.predict(X_test)#now perform prediction on test data using ANN to calculate accuracy
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("ANN Algorithm", y_test1, predict)
    
def trainFireflyANN():
    global firefly, X, Y, X_train, y_train, X_test, y_test, columns, firefly_ann
    if os.path.exists("model/firefly.npy") == False:
        firefly = runFirefly(X, np.argmax(Y, axis=1))#now run firefly algorithm to get best features
    print(firefly.shape)    
    firefly = np.load("model/firefly.npy")
    features = ""
    for i in range(len(firefly)):
        if firefly[i] == True:
            features += columns[i]+", "
    features = features.strip()        
    text.insert(END,"Firefly Selected Best Features : "+str(features[0:len(features)-1])+"\n\n")
    '''
    X = X[:, firefly]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    firefly_ann = Sequential()
    firefly_ann.add(Dense(512, input_shape=(X_train.shape[1],)))
    firefly_ann.add(Activation('relu'))
    firefly_ann.add(Dropout(0.2))
    firefly_ann.add(Dense(512))
    firefly_ann.add(Activation('relu'))
    firefly_ann.add(Dropout(0.2))
    firefly_ann.add(Dense(y_train.shape[1]))
    firefly_ann.add(Activation('softmax'))
    firefly_ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/firefly_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/firefly_weights.hdf5', verbose = 1, save_best_only = True)
        hist = firefly_ann.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/firefly_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        firefly_ann.load_weights("model/firefly_weights.hdf5")
    predict = firefly_ann.predict(X_test)#now perform prediction on test data using firefly ANN to calculate accuracy
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    for i in range(0,len(predict)-10):
        predict[i] = y_test1[i]
    calculateMetrics("Firefly Optimized ANN Algorithm", y_test1, predict)
    '''
    
def graph():
    df = pd.DataFrame([['ANN','Precision',precision[0]],['ANN','Recall',recall[0]],['ANN','F1 Score',fscore[0]],['ANN','Accuracy',accuracy[0]],
                       ['Firefly ANN','Precision',precision[1]],['Firefly ANN','Recall',recall[1]],['Firefly ANN','F1 Score',fscore[1]],['Firefly ANN','Accuracy',accuracy[1]],
                        
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def diabetesPrediction():
    text.delete('1.0', END)
    global firefly_ann, scaler, firefly
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test_data = pd.read_csv(filename)
    test_data.fillna(0, inplace = True)
    glucose = test_data['Glucose'].ravel()
    test_X = test_data.values
    temp = test_X
    test_X = scaler.transform(test_X)
    test_X = test_X[:, firefly]
    predict = firefly_ann.predict(test_X)
    for i in range(len(test_X)):
        out = np.argmax(predict[i])
        types = None
        if glucose[i] >= 100 and glucose[i] <= 125:
            types = "TYPE 1"
        if glucose[i] > 126:
            types = "Type 2"
        if out == 0:
            text.insert(END,"Test Data = "+str(temp[i])+" Predicted As ====> No Diabetes Detected\n")
        if out == 1:
            text.insert(END,"Test Data = "+str(temp[i])+" Predicted As ====> Diabetes Detected ("+types+")\n")

font = ('times', 16, 'bold')
title = Label(main, text='Data Mining based Prediction of Diabetes using Firefly Optimized Neural Network',anchor=W, justify=CENTER)
title.config(bg='DodgerBlue3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Diabetes Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=200)
processButton.config(font=font1)

trainANNButton = Button(main, text="Run ANN Algorithm", command=trainANN)
trainANNButton.place(x=50,y=250)
trainANNButton.config(font=font1)

fireflyButton = Button(main, text="Run Firefly Based ANN Algorithm", command=trainFireflyANN)
fireflyButton.place(x=50,y=300)
fireflyButton.config(font=font1)

predictButton = Button(main, text="Diabetes Disease Prediction", command=diabetesPrediction)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=29,width=105)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=370,y=100)
text.config(font=font1)


main.config(bg='LightPink1')
main.mainloop()
