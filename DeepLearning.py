import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import ModelCheckpoint

root=r'data\EmmetSamples'
SA='Emmet'
t1={}
for year in range(2021,2024):
    df=pd.read_csv(root+'/'+SA+'Samples'+str(year)+'.csv')
    if SA=='Wangkui':
        df=df[df['Class']<4]
        n_classes=3
    else:
        df=df[df['Class']>0]
        n_classes=2
    feat_labels=list(df.columns)[3:]
    df=df[['Lon','Lat','Class']+feat_labels].values
    train_data,test_data=train_test_split(df,test_size=0.2)
    t1[str(year)]=[train_data,test_data]
all_data=[]
feature_num=10
n_timesteps=7
for i in range(2021,2024):
    for j in range(2021,2024):
        t={}
        t['train_year']=i
        t['test_year']=j
        train_data=t1[str(i)][0]
        num=train_data.shape[0]
        t['train_x']=train_data[:,3:].reshape((num,n_timesteps,feature_num))
        t['train_y']=train_data[:,2]
        test_data=t1[str(j)][1]
        num=test_data.shape[0]
        t['test_x']=test_data[:,3:].reshape((num,n_timesteps,feature_num))
        t['test_y']=test_data[:,2]
        all_data.append(t)
model1 = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_timesteps, feature_num)),
    MaxPooling1D(pool_size=1),
    Dropout(0.5),
    Conv1D(filters=128, kernel_size=1, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(n_classes, activation='softmax')
],name='1D_CNN')
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2 = Sequential([
    LSTM(units=64,activation='relu', input_shape=(n_timesteps, feature_num),return_sequences=True),
    Dropout(0.5),
    LSTM(units=128,activation='relu',return_sequences=False),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
],name='LSTM')
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
models=[model1,model2]

for model in models:
    t=[]
    for data in all_data:
        x_train=data['train_x']
        y_train=data['train_y']-1
        x_test=data['test_x']
        y_test=data['test_y']-1
        print(data['train_year'],data['test_year'])
        t1=data['train_year']
        t2=data['test_year']
        mc=ModelCheckpoint('best.h5',save_best_only=True,monitor='val_accuracy')
        epoch=300
        model.fit(x_train, y_train,epochs=epoch,batch_size=32,verbose=1, validation_data=(x_test,y_test),callbacks=mc)
        if t1==t2:
            model.load_weights('best.h5')
        predictions = model.predict(x_test)
        predictions=np.argmax(predictions,axis=1)
        accuracy = accuracy_score(y_test, predictions)
        print(f'{model.name}: Accuracy = {accuracy:.4f}')






