import numpy as np #Allows numpy arrays to split data into columns
import pandas as pd #Allows data processing
from sklearn.preprocessing import LabelEncoder #Enabled label encoding
from sklearn.model_selection import train_test_split #Allows splitting of data into testing and training
from sklearn.preprocessing import StandardScaler #Allows scaling
from sklearn.preprocessing import LabelEncoder #Enabled label encoding
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score,recall_score,precision_score,f1_score
import pickle
import glob

#-----Algorithms-----
#from sklearn.linear_model import SGDRegressor #To train model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression #To train model
from sklearn.neighbors import KNeighborsClassifier #To train model
from sklearn.ensemble import RandomForestClassifier #To train model

def create_new_training_dataframe():
    # bookingID: trip 
    #- accuracy: accuracy inferred by GPS in meters
    #- bearing: GPS bearing in degree
    #- acceleration_x: accelerometer reading at x axis (m/s2)
    #- acceleration_y: accelerometer reading at y axis (m/s2)
    #- acceleration_z: accelerometer reading at z axis (m/s2)
    #(Acceleration determines the acceleration / vibration of the device in motion. Each of the axis can be thought of as a different sensor even though they reside on the same physical chip)
    #- gyro_x: gyroscope reading in x axis (rad/s)
    #- gyro_y: gyroscope reading in y axis (rad/s)
    #- gyro_z: gyroscope reading in z axis (rad/s)
    #(Gyroscope data determine the orientation of the device relative to earth's gravity)
    #- second: time of the record by number of seconds
    #- speed: speed measured by GPS in m/s
    #- label: tags to indicate dangerous driving trip (0: Normal trip / 1: Dangerous trip)  
    # Read feature csv files and concate as single dataframe
    path_f = r'./datasets' # use your path
    all_feature_files = glob.glob(path_f + "/*.csv")
    li = []
    for filename in all_feature_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df_f = pd.concat(li, axis=0, ignore_index=True)
    df_f=pd.read_csv('./datasets/training_dataset.csv')
    df_l = pd.read_csv('./label/training_labels.csv', index_col=None, header=0)
    # Sort dataframe according to bookingID and second (time)
    df_f = df_f.sort_values(by=['bookingID', 'second'], ascending=[True, True])
    df_l = df_l.sort_values(by='bookingID', ascending=True)
    df = pd.merge(left=df_f, right=df_l, how='inner', left_on='bookingID', right_on='bookingID')
    df=df.reset_index(drop=True)
    final_df=pd.DataFrame(columns=['bookingID','Accuracy','Bearing','acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','second','Speed','label'])
    temp=pd.DataFrame()
    for i in df['bookingID'].unique():
        temp['bookingID']=i
        temp=df.loc[df['bookingID']==i].mean()
        if(temp['label']==0.5):
            temp['label']=1
        elif(temp['label']<0.5):
            temp['label']=0
        final_df=final_df.append(temp,ignore_index=True)
    #Put the output into a new csv
    final_df.to_csv('newdataframe.csv',index=False)
    print("new dataset created")

def training_model(model_selection):
    df=pd.read_csv('./datasets/newdataframe.csv')
    copy_df=df.copy()
    df=df.drop(columns=['bookingID','second'])
    target = 'label'
    features = ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y', 'acceleration_z','gyro_x','gyro_y','gyro_z','Speed'] #For the everything else
    x = df.loc[:, features] #Your input. Take all row and all columns in features
    y = df.loc[:, target] #Take  all row and only survived column
    #Split data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25) #You would want to reshuffle your data
    x_scaler = StandardScaler() 
    z_train = x_scaler.fit_transform(x_train)
    z_train = pd.DataFrame(data = z_train, columns = features ) #input data is z_train, specified columns
    if(model_selection=='SGDClassifier'):
        model = SGDClassifier() #Non-optimised
        model.fit(x_train,y_train)
        pickle.dump(model,open('./models/SGDClassifier.p','wb'))
    elif(model_selection=='SGDClassifier_scaled'):
        model = SGDClassifier() #Non-optimised
        model.fit(x_train,y_train) #Training scaled x to y
        pickle.dump(model,open('./models/SGDClassifier_scaled.p','wb'))
    elif(model_selection=='LogisticRegressor'):
        model = LogisticRegression() #No parameter tuning
        model.fit(x_train,y_train)
        pickle.dump(model,open('./models/LogisticRegressor.p','wb'))
    elif(model_selection=='LogisticRegressor_scaled','wb'):
        model = LogisticRegression() #No parameter tuning
        model.fit(x_train,y_train)
        pickle.dump(model,open('./models/LogisticRegressor_scaled.p','wb'))
    elif(model_selection=='KNNClassifier'):
        model = KNeighborsClassifier()
        model.fit(x_train,y_train)
        pickle.dump(model,open('./models/KNNClassifier.p','wb'))
    elif(model_selection=='KNNClassifier_Scaled'):
        model = KNeighborsClassifier()
        model.fit(z_train,y_train)
        pickle.dump(model,open('./models/KNNClassifier_scaled.p','wb'))
    elif(model_selection=='RandomForest'):
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        pickle.dump(model,open('./models/RandomForest.p','wb'))
    elif(model_selection=='RandomForest_scaled'):
        model = RandomForestClassifier()
        model.fit(z_train,y_train)
        pickle.dump(model,open('./models/RandomForest_scaled.p','wb'))
    print("Training completed")

def prediction(model_selection,data):
    model=pickle.load('./models/'+model_selection+'.p','rb')
    result=model.predict(data)
    data=data.append(result)
    print("Prediction returned, appened to end of data")
    return(data)
