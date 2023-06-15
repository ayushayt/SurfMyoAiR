import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from scipy import signal
import scipy.io
from scipy import signal
from scipy import interpolate
import mne
from sklearn.metrics import accuracy_score,confusion_matrix
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Input,UpSampling1D
from tensorflow.keras.layers import concatenate, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import zscore
from scipy.signal import butter, lfilter
import time
import cv2


def process_EMG(trial,duration,width,interp_type):

    X = np.zeros((2600,64,64,5))
    Y = np.zeros((2600,))
    testctr = 0
    
    for folder in os.listdir('EMG_Data'):
        for rep in range (1,3):
            for num in range(65,91):
                emg_sig = np.load('EMG_Data/'+folder+'/Segmented/EMG_'+chr(num)+'_'+str(trial+1)+'_'+str(rep)+'.npy')
                
                emg_signal = np.zeros((duration,5))

                
                if len(emg_sig)<=duration:
                    for i in range(0,5):
                        x = np.linspace(0, 1,len(emg_sig))
                        y = emg_sig[:,i]
                        f = interpolate.interp1d(x, y,kind=interp_type)
                        xnew = np.linspace(0, 1,duration)
                        emg_signal[:,i] = f(xnew)

                else:
                    emg_signal = emg_sig[0:duration,:]
                
                for i in range(0,5):
                    fs = 2000
                    w = 6.
                    freq = np.linspace(1, fs/2, width)
                    widths = w*fs / (2*freq*np.pi)
                    cwtm = signal.cwt(emg_signal[:,i], signal.morlet2, widths, w=w)
                    cwtm = np.abs(cwtm)
                    cwtm = cv2.resize(cwtm, (64,64))
                    cwtm = (cwtm-np.mean(cwtm))/np.std(cwtm)
                    X[testctr,:,:,i] = cwtm
                    
                Y[testctr] = num-65
                #print(num)
                testctr = testctr + 1

    return X,Y

folderarr = []
for fo in os.listdir('EMG_Data'):
    folderarr.append(fo)
    

    
accarr = []

for fold_no in range(0,5):

    trainrefctr = 0
    testrefctr = 0

    
    X_train = np.zeros((10400,64,64,5))
    X_test = np.zeros((2600,64,64,5))
    Y_train = np.zeros((10400,))
    Y_test = np.zeros((2600,))

    print('Fold number : ' + str(fold_no))

    for fold_id in range(0,5):
        if fold_id==fold_no:
            [X,Y] = process_EMG(fold_id,8000,60,'cubic') 
            X_test[int(testrefctr*2600):int((testrefctr+1)*2600),:,:,:] = X
            Y_test[int(testrefctr*2600):int((testrefctr+1)*2600)] = Y
            testrefctr = testrefctr + 1
            
        else:
            [X,Y] = process_EMG(fold_id,8000,60,'cubic') 
            X_train[int(trainrefctr*2600):int((trainrefctr+1)*2600),:,:,:] = X
            Y_train[int(trainrefctr*2600):int((trainrefctr+1)*2600)] = Y
            trainrefctr = trainrefctr + 1
                

    
    np.save('cwt/X_train_cwt_userdep_fold_'+str(fold_no)+'.npy',X_train)
    np.save('cwt/Y_train_cwt_userdep_fold_'+str(fold_no)+'.npy',Y_train)
    np.save('cwt/X_test_cwt_userdep_fold_'+str(fold_no)+'.npy',X_test)
    np.save('cwt/Y_test_cwt_userdep_fold_'+str(fold_no)+'.npy',Y_test)

    tensorflow.keras.backend.clear_session()
    i1 = Input(shape=(64,64,5))
    x1 = Conv2D(32,(3,3),padding='same',activation='relu')(i1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(64,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(128,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(256,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Flatten()(x1)
    x1 = Dropout(0.5)(x1)
    output = Dense(26, activation='softmax')(x1)
    
    model = Model(inputs=i1, outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=10)
    model.fit(X_train, y=to_categorical(Y_train),validation_split=0.2,epochs=500, batch_size=256,verbose=0,callbacks=[es])
    
    pred = model.predict(X_test)
    Y_pred = np.argmax(pred,axis=1)
    print(accuracy_score(Y_pred,Y_test)*100)
    accarr.append(accuracy_score(Y_pred,Y_test)*100)
            
print(np.mean(accarr)) 


