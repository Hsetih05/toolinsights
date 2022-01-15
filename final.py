from matplotlib.pyplot import plot
import streamlit as st

import plotly_express as px
import pandas as pd 
st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.subheader("File Uploader")
uploaded_file = st.sidebar.file_uploader(label="upload your csv file or excel file.", type = ['csv', 'xlsx'])
st.title('Tool Insights')
original_list = ['Home','Tool Wear Detection', 'Tool Wear Prediction', 'Tool Performance', 'Outlier Detection','Tool Detection','Data Analysis','Tips',]
result = st.selectbox('Select Your Option', original_list)
st.write(f'Selected Option: ',(result))
if result == 'Home':
    st.subheader('Home')
    global df
    if uploaded_file is not None:
        st.write(uploaded_file)

    else:
        st.subheader("Please upload file to the application")
        st.markdown('''
        ##### And Use Various Features :
        + ##### Tool Wear Detection.
        + ##### Tool Wear Prediction.
        + ##### Tool Performanace
        + ##### Outlier Detection.
        + ##### Tool Detection.
        + ##### Data Analysis.
        
        
        
        
        ''')



if result == 'Tool Detection':
    st.subheader('Tool Detection')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow
    from tensorflow import keras

    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    train=pd.read_csv('E:/hset/New folder/train.csv')
    train.shape
    train.columns

    train.head()
    plt.style.use('seaborn-darkgrid')
    fig, ((ax0,ax1,ax2),(ax3,ax4,ax5))=plt.subplots(nrows=2,
                                       ncols=3,
                                       figsize=(30, 20))
    def label_function(val):
        return f'{val / 100 * len(train):.0f}\n{val:.0f}%'

    # feedrate
    train.groupby('material').size().plot(kind='pie',
                                      autopct=label_function, 
                                      textprops={'fontsize': 15},
                                      ax=ax0)                                         
    ax0.set_xlabel('material',size=15)

    # Tool Condition
    train.groupby('tool_condition').size().plot(kind='pie', 
                                              autopct=label_function,
                                              textprops={'fontsize': 15},
                                              ax=ax1)
    ax1.set_xlabel('Tool Condition',size=15)
    # Pressure
    train.groupby('clamp_pressure').size().plot(kind='pie', 
                                      autopct=label_function, 
                                      textprops={'fontsize': 15},
                                      colors=['violet', 'lime','tomato'],
                                      ax=ax2)
    ax2.set_xlabel('Clamp Pressure',size=15)

    # Machining Finalized
    train.groupby('machining_finalized').size().plot(kind='pie',
                                                 autopct=label_function, 
                                                 textprops={'fontsize': 15},
                                                 colors=['tomato', 'gold','violet','lime'],
                                                 ax=ax3)
    ax3.set_xlabel('Machining_finalized',size=15)

    # passed_visual_inspection
    train.groupby('passed_visual_inspection').size().plot(kind='pie',
                                                 autopct=label_function, 
                                                 textprops={'fontsize': 15},
                                                 ax=ax4)                                         
    ax4.set_xlabel('passed_visual_inspection',size=15)
    # feedrate
    train.groupby('feedrate').size().plot(kind='pie',
                                      autopct=label_function, 
                                      textprops={'fontsize': 15},
                                      ax=ax5)                                         
    ax5.set_xlabel('feedrate',size=15)

    # showing the figure
    st.pyplot(fig)


    frames=list()
    for i in range(1,19):
        exp = '0' + str(i) if i < 10 else str(i)
        frame = pd.read_csv("E:/hset/New folder/experiment_{}.csv".format(exp))
        row = train[train['No'] == i]
        frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
        frames.append(frame)
    df = pd.concat(frames, ignore_index = True)

    def dummy_creation(dataset,dummy_categories):
        for i in dummy_categories:
            dataset_dummy=pd.get_dummies(dataset[i])
            dataset=pd.concat([dataset,dataset_dummy],
                      axis=1)
            dataset=dataset.drop(i,axis=1)
        return dataset
    df['Machining_Process'].unique()
    df=dummy_creation(df, ['Machining_Process'])
    df.info()
    # Seperating features and labels
    features=df.drop('target',axis="columns")
    labels=df['target']

    from sklearn.model_selection import train_test_split
    x_Train, x_Test, y_Train, y_Test=train_test_split(features,labels,test_size=0.2)
    st.write("--------------------------------------------")

    np.random.seed(42)
    from sklearn.ensemble import RandomForestClassifier
    random_forest=RandomForestClassifier()
    st.write(random_forest.get_params())

    random_forest.fit(x_Train,y_Train)
    random_forest_acc=random_forest.score(x_Test,y_Test)
    y_pred=random_forest.predict(x_Test)
    accuracy_score(y_true=y_Test,
               y_pred=y_pred)
    
    confusion_matrix(y_true=y_Test,y_pred=y_pred)
    for n_estimator in range(10,100,10):
        clf=RandomForestClassifier(n_estimators=n_estimator)
        clf.fit(x_Train,y_Train)
        y_pred=clf.predict(x_Test)
        if (accuracy_score(y_Test,y_pred) > random_forest_acc):
            st.write("---------------------------------------")
            st.write(f"{n_estimator} is the n_estimator")
            st.write(f"Model Accuracy:{accuracy_score(y_Test,y_pred)}")
            st.write("------------------------------------")  


    random_forest=RandomForestClassifier(n_estimators=30)
    random_forest.fit(x_Train,y_Train)
    y_pred=random_forest.predict(x_Test)
    new_random_forest_acc=accuracy_score(y_true=y_Test,y_pred=y_pred)
    print(new_random_forest_acc-random_forest_acc)


    from sklearn.tree import DecisionTreeClassifier
    decision_trees=DecisionTreeClassifier()
    st.write(decision_trees.get_params())

    st.write("---------------------------------------")

    decision_trees.fit(x_Train,y_Train)
    decision_trees.score(x_Test,y_Test)
    import xgboost as xgb
    from xgboost import XGBClassifier
    xgboost_clf=XGBClassifier()
    st.write(xgboost_clf.fit(x_Train,y_Train))
    st.write(xgboost_clf.score(x_Test,y_Test))



if result == 'Tool Wear Prediction':
    st.subheader('Tool Wear Prediction')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.linear_model import  LogisticRegression
    from sklearn import svm
    from sklearn import tree
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.linear_model import SGDClassifier

    train=pd.read_csv('E:/hset/New folder/train.csv')

    le1=LabelEncoder()
    le1.fit(train['material'])
    train['Encoded_material']=le1.transform(train['material'])

    le2=LabelEncoder()
    le2.fit(train['tool_condition'])
    train['Encoded_tool_condition']=le2.transform(train['tool_condition'])

    le3=LabelEncoder()
    le3.fit(train['machining_finalized'])
    train['Encoded_machining_finalized']=le3.transform(train['machining_finalized'])

    le4=LabelEncoder()
    le4.fit(train['feedrate'])
    train['Encoded_feedrate']=le4.transform(train['feedrate'])

    le5=LabelEncoder()
    le5.fit(train['clamp_pressure'])
    train['Encoded_clamp_pressure']=le5.transform(train['clamp_pressure'])

    train.drop(['passed_visual_inspection','tool_condition','material','machining_finalized'],axis=1,inplace=True)

    plt.figure(figsize=(20,18))
    sns.heatmap(train.corr(),linewidths=.1,annot=True)
    st.pyplot(plt)

    frames = []
    for i in range(1,19):
        ex_num = '0' + str(i) if i < 10 else str(i)
        frame = pd.read_csv("E:/hset/New folder/experiment_{}.csv".format(ex_num))

        ex_result_row = train[train['No'] == i]
        
        le6=LabelEncoder()
        le6.fit(frame['Machining_Process'])
        frame['Encoded_Machining_Process']=le6.transform(frame['Machining_Process'])

        frame['Encoded_feedrate'] = ex_result_row.iloc[0]['Encoded_feedrate']
        frame['Encoded_clamp_pressure'] = ex_result_row.iloc[0]['Encoded_clamp_pressure']
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    df.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia'],axis=1,inplace=True)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.subheader(">>>>> Heatmap of frames after drop column <<<<<")
    plt.figure(figsize=(50,50))
    sns.heatmap(df.corr(),linewidths=.1,annot=True)
    st.pyplot(plt)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    corr = df.corr()
    st.write("\n>>>>> All features Correlation shape <<<<<")
    st.write(df.corr().shape)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    
    st.subheader("\n>>>>> Encoded_feedrate abs Correlation > 0.3 <<<<<")
    cor_feedrate = abs(corr["Encoded_feedrate"])
    relevant_features = cor_feedrate[cor_feedrate>0.3]
    st.write(relevant_features)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    newDf_feedrate = df[['X1_ActualPosition', 'X1_CommandPosition', 'Y1_ActualPosition', 'Y1_CommandPosition', 'Z1_ActualPosition', 'Z1_CommandPosition', 'S1_ActualVelocity', 'S1_CommandVelocity', 'S1_CurrentFeedback', 'S1_DCBusVoltage', 'S1_OutputVoltage', 'S1_OutputPower', 'M1_sequence_number', 'M1_CURRENT_FEEDRATE', 'Encoded_feedrate']]
    st.write("\n---------- newDf feedrate info: ----------")
    st.write(newDf_feedrate.info())
    st.write("---------- newDf feedrate head: ----------")
    st.write(newDf_feedrate.head(3))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("\n>>>>> Encoded_clamp_pressure abs Correlation > 0.1 <<<<<")
    cor_clamp_pressure = abs(corr["Encoded_clamp_pressure"])
    st.write("--------------------------------------------------------------------------------------------------------")
    #Selecting highly correlated features
    relevant_features2 = cor_clamp_pressure[cor_clamp_pressure>0.1]
    st.write(relevant_features2)
    st.write("--------------------------------------------------------------------------------------------------------")
    newDf_pressure = df[['X1_ActualPosition', 'X1_CommandPosition', 'X1_OutputCurrent', 'Y1_ActualPosition', 'Y1_CommandPosition', 'Y1_OutputCurrent', 'Z1_ActualPosition', 'Z1_CommandPosition', 'S1_ActualVelocity', 'S1_CommandVelocity', 'S1_OutputVoltage', 'M1_CURRENT_FEEDRATE', 'Encoded_clamp_pressure']]
    st.write("\n---------- newDf pressure info: ----------")
    st.write(newDf_pressure.info())
    st.write("---------- newDf pressure head: ----------")
    st.write(newDf_pressure.head(3))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    y=newDf_feedrate['Encoded_feedrate']
    X=newDf_feedrate.drop(['Encoded_feedrate'], axis=1)
    st.write("\n========== y.newDf_feedrate(3) ==========")
    st.write(y.head(3))
    st.write("\n========== X.newDf_feedrate(3) ==========")
    st.write(X.head(3))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    LR_feedrate = LogisticRegression(C=1e20)
    LR_feedrate.fit(X_train, y_train)
    st.write("\n========== Logistic Regression score of feedrate ==========")
    st.write(LR_feedrate.score(X_test,y_test))
    st.write("--------------------------------------------------------------------------------------------------------")

    st.write("--------------------------------------------------------------------------------------------------------")
    SVM_feedrate = svm.SVC()
    SVM_feedrate.fit(X_train, y_train)
    st.write("\n========== Support vector machine score of feedrate  ==========")
    st.write(SVM_feedrate.score(X_test, y_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    mlp_feedrate = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    mlp_feedrate.fit(X_train, y_train)
    st.write("\n========== MLPClassifier score of feedrate  ==========")
    st.write(mlp_feedrate.score(X_test, y_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    nrc_feedrate= NearestCentroid()
    nrc_feedrate.fit(X_train, y_train)
    st.write("\n========== NearestCentroid score of feedrate  ==========")
    st.write(nrc_feedrate.score(X_test, y_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    SGD_feedrate= SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    SGD_feedrate.fit(X_train, y_train)
    st.write("\n========== SGDClassifier score of feedrate  ==========")
    st.write(SGD_feedrate.score(X_test, y_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    tree_feedrate = tree.DecisionTreeClassifier()
    tree_feedrate.fit(X_train, y_train)
    st.write("\n========== DecisionTreeClassifier score of feedrate  ==========")
    st.write(tree_feedrate.score(X_test, y_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    predictions_feedrate = tree_feedrate.predict(X_test)
    st.write("\n========== Prediction_feedrate results  ==========")
    st.write(predictions_feedrate)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    y1=newDf_pressure['Encoded_clamp_pressure']
    X1=newDf_pressure.drop(['Encoded_clamp_pressure'], axis=1)
    st.write("\n========== y.head(3) ==========")
    st.write(y1.head(3))
    st.write("\n========== X.head(3) ==========")
    st.write(X1.head(3))
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=7)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    LR_pressure = LogisticRegression(C=1e20)
    LR_pressure.fit(X1_train, y1_train)
    st.write("\n========== Logistic Regression score of pressure ==========")
    st.write(LR_pressure.score(X1_test,y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    SVM_pressure = svm.SVC()
    SVM_pressure.fit(X1_train, y1_train)
    st.write("\n========== Support vector machine score of pressure  ==========")
    st.write(SVM_pressure.score(X1_test, y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    mlp_pressure = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    mlp_pressure.fit(X1_train, y1_train)
    st.write("\n========== MLPClassifier score of pressure  ==========")
    st.write(mlp_pressure.score(X1_test, y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    nrc_pressure= NearestCentroid()
    mlp_pressure.fit(X1_train, y1_train)
    st.write("\n========== NearestCentroid score of pressure  ==========")
    st.write(mlp_pressure.score(X1_test, y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    SGD_pressure= SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    SGD_pressure.fit(X1_train, y1_train)
    st.write("\n========== SGDClassifier score of pressure  ==========")
    st.write(SGD_pressure.score(X1_test, y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    tree_pressure = tree.DecisionTreeClassifier()
    tree_pressure.fit(X1_train, y1_train)
    st.write("\n========== DecisionTreeClassifier score of pressure  ==========")
    st.write(tree_pressure.score(X1_test, y1_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    predictions_pressure = tree_pressure.predict(X1_test)
    st.write("\n========== Prediction_pressure results  ==========")
    st.write(predictions_pressure)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    finalDf = pd.DataFrame(columns=['feedrate','clampPressure'])
    finalDf['feedrate'] = predictions_feedrate.tolist()
    finalDf['clampPressure'] = predictions_pressure.tolist()
    st.write("----- finalDf.info() -----")
    st.write(finalDf.info())
    st.write("----- finalDf.head(3) -----")
    st.write(finalDf.head(3))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    y_final=train['Encoded_tool_condition']
    X_final=train[['Encoded_feedrate', 'Encoded_clamp_pressure']]
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X_final, y_final, test_size=0.3, random_state=7)

    LR_tool_condition = LogisticRegression(C=1e20)
    LR_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== LogisticRegression score of tool_condition  ==========")
    st.write(LR_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    SVM_tool_condition = svm.SVC()
    SVM_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== Support vector machine score of tool_condition  ==========")
    st.write(SVM_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    mlp_tool_condition = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    mlp_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== MLPClassifier score of tool_condition  ==========")
    st.write(mlp_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")


    nrc_tool_condition= NearestCentroid()
    nrc_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== NearestCentroid score of tool_condition  ==========")
    st.write(nrc_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    SGD_tool_condition= SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    SGD_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== SGDClassifier score of tool_condition  ==========")
    st.write(SGD_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    tree_tool_condition = tree.DecisionTreeClassifier()
    tree_tool_condition.fit(X_final_train, y_final_train)
    st.write("\n========== DecisionTreeClassifier score of tool_condition  ==========")
    st.write(tree_tool_condition.score(X_final_test, y_final_test))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")
    predictions_tool_condition = SGD_tool_condition.predict(X_final_test)
    st.write("\n========== predictions_tool_condition results  ==========")
    st.write(predictions_tool_condition)
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")

    pred=SGD_tool_condition.predict(finalDf)
    output=pd.DataFrame({'Feedrate':finalDf['feedrate'], 'Clamp_Pressure':finalDf['clampPressure'], 'Tool_Condition':pred})
    st.write(output.head(5))
    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")


    finalDf['feedrate'] = finalDf['feedrate'].apply(np.sign).replace({0.0: 3, 0.1: 6, 0.2: 12, 0.3: 15, 0.4:20})
    finalDf['clampPressure'] = finalDf['clampPressure'].apply(np.sign).replace({0.0: 2.5, 0.1: 3.0, 0.2: 4.0})
    pred = pred.astype(str)
    for i in range(len(pred)):
        if (pred[i]=='0'):
            pred[i]='unworn'
        elif (pred[i]=='1'):
            pred[i]='worn'

    output=pd.DataFrame({'Feedrate':finalDf['feedrate'], 'Clamp_Pressure':finalDf['clampPressure'], 'Tool_Condition':pred})
    st.write(output.head(5))

    st.write("--------------------------------------------------------------------------------------------------------")
    st.write("--------------------------------------------------------------------------------------------------------")


if result == 'Outlier Detection':
    st.subheader('Outlier Detection')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import os
    for dirname, _, filenames in os.walk('E:/hset/New folder/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main_df=pd.read_csv('E:/hset/New folder/train.csv')  
    main_df=main_df.fillna('no')
    main_df.head()
    import glob
    #set working directory
    os.chdir('E:/hset/New folder/')
    files = list()

    for i in range(1,19):
        exp_number = '0' + str(i) if i < 10 else str(i)
        file = pd.read_csv("E:/hset/New folder/experiment_{}.csv".format(exp_number))
        row = main_df[main_df['No'] == i]
    
        #add experiment settings to features
        file['feedrate']=row.iloc[0]['feedrate']
        file['clamp_pressure']=row.iloc[0]['clamp_pressure']
    
        # Having label as 'tool_conidtion'
    
        file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
        files.append(file)
        df = pd.concat(files, ignore_index = True)
        df.head()
    
    df.shape
    # Convert 'Machining_process' into numerical values
    pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'end':10,'Starting':11}

    data=[df]

    for dataset in data:
        dataset['Machining_Process']=dataset['Machining_Process'].map(pro)
    
    df=df.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia'],axis=1)
    corm=df.corr()
    corm

    #checking the relationship between the variables by applying the correlation 
    plt.figure(figsize=(30, 25))
    p = sns.heatmap(df.corr(), annot=True)
    
    X=df.drop(['label','Machining_Process'],axis=1)
    Y=df['label']
    st.write("............................................................")
    st.write('The dimension of X table is: ',X.shape,'\n')
    st.write('The dimension of Y table is: ', Y.shape)
    st.write("............................................................")

    from sklearn.model_selection import train_test_split

    #divided into testing and training
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    from sklearn import linear_model
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB

    sgd_model=SGDClassifier()
    sgd_model.fit(x_train,y_train)
    st.write("............................................................")
    st.write("SGD Pred")
    sgd_model_pred=sgd_model.predict(x_test)
    acc_sgd_model=round(sgd_model.score(x_train, y_train)*100,2)
    acc_sgd_model
    st.write("............................................................")
    

    st.write("............................................................")

    rmf_model=RandomForestClassifier()
    rmf_model.fit(x_train,y_train)
    st.write("RMF Pred")

    rmf_model_pred=rmf_model.predict(x_test)
    acc_rmf_model=round(rmf_model.score(x_train, y_train)*100,2)
    acc_rmf_model
    st.write("............................................................")
    st.write("............................................................")
    st.write("Logistic Regression")
    log_reg=LogisticRegression()
    log_reg.fit(x_train,y_train)
    log_reg_pred=log_reg.predict(x_test)
    acc_log_reg=round(log_reg.score(x_train,y_train)*100,2)
    acc_log_reg
    st.write("............................................................")
    
    st.write("............................................................")
    st.write("Linear SVM")
    svm_model=LinearSVC()
    svm_model.fit(x_train,y_train)
    svm_model_pred=svm_model.predict(x_test)
    acc_svm_model=round(svm_model.score(x_train,y_train)*100,2)
    acc_svm_model
    st.write("............................................................")

    results = pd.DataFrame({
        'Model': ['Support Vector Machines', 'Logistic Regression', 
                'Random Forest','Stochastic Gradient Decent'],
        'Score': [acc_svm_model, acc_log_reg, 
                acc_rmf_model,acc_sgd_model]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Score')
    result_df

    from sklearn.model_selection import cross_val_score
    rmf_model = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(rmf_model, x_train, y_train, cv=10, scoring = "accuracy")
    st.write("............................................................")
    st.write("Scores:", scores,'\n')
    st.write("Mean:", scores.mean(),'\n')
    st.write("Standard Deviation:", scores.std())
    st.write("............................................................")
    rmf_model = RandomForestClassifier(n_estimators=100, oob_score = True)
    rmf_model.fit(x_train, y_train)
    y_prediction = rmf_model.predict(x_test)

    rmf_model.score(x_train, y_train)
    st.write("............................................................")
    st.write("Random Forest Classfier Score")
    acc_rmf_model = round(rmf_model.score(x_train, y_train) * 100, 2)
    st.write(round(acc_rmf_model,2,), "%")
    st.write("oob score:", round(rmf_model.oob_score_, 4)*100, "%")
    st.write("............................................................")
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    from sklearn.model_selection import cross_val_predict

    predictions = cross_val_predict(rmf_model, x_train, y_train, cv=3)
    predictions[:10] # first 10 predictions

    confusion_matrix(y_train,predictions)
    st.write("............................................................")

    st.write("Precision_score: ", precision_score(y_train,predictions),'\n')
    st.write("Recall: ", recall_score(y_train,predictions),'\n')
    st.write("Accruacy_score: ", accuracy_score(y_train,predictions),'\n')
    st.write("F_score: ", f1_score(y_train, predictions))
    st.write("............................................................")

    from sklearn.metrics import precision_recall_curve

    # getting the probabilities of our predictions
    y_scores = rmf_model.predict_proba(x_train)
    y_scores = y_scores[:,1]

    precision, recall, threshold = precision_recall_curve(y_train, y_scores)
    def plot_precision_and_recall(precision, recall, threshold):
        plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
        plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
        plt.xlabel("threshold", fontsize=19)
        plt.legend(loc="upper right", fontsize=19)
        plt.ylim([0, 1])

    plt.figure(figsize=(14, 7))
    plot_precision_and_recall(precision, recall, threshold)
    st.pyplot(plt)



if result == 'Tool analysis':
    st.subheader('Tool analysis')

    from pprint import pprint

    import pandas as pd 
    import numpy as np

    # Standard plotly imports
    import plotly.graph_objs as go
    import plotly.figure_factory as pff
    from plotly.subplots import make_subplots

    import cufflinks
    cufflinks.go_offline(connected=True)

    import matplotlib.pyplot as plt

    from sklearn.metrics import matthews_corrcoef
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    data_dir = 'E:/hset/New folder/'
    outcomes = pd.read_csv(data_dir + 'train.csv')
    outcomes.info()
    
    part_out = outcomes[outcomes['passed_visual_inspection'].notna()]
    pprint(pd.crosstab(part_out.tool_condition, part_out.passed_visual_inspection))

    passed = part_out.passed_visual_inspection.eq('yes').mul(1)
    wear = part_out.tool_condition.eq('unworn').mul(1)

    st.write('\nPearson correlation coefficient: {:.2f}'.format(matthews_corrcoef(passed, wear)))
    experiment1 = pd.read_csv(data_dir + 'experiment_01.csv')
    experiment1.reset_index(inplace=True)
    experiment1.info()

    experiment1.head()

    (experiment1[['Machining_Process', 'index']].groupby('Machining_Process').count())

    # Prep data functions
    def clean_data(data):
        """Use this function to keep only CNC active cutting actions"""
        keep_act = ['Layer 1 Down', 'Layer 1 Up', 'Layer 2 Down', 'Layer 2 Up', 'Layer 3 Down', 'Layer 3 Up']
        data = data[data['Machining_Process'].isin(keep_act)]
        #print(data[['Machining_Process', 'index']].groupby('Machining_Process').count())
    
        data.drop('Machining_Process', inplace=True, axis=1)
        return data

    def scale_decompose(data, pca_n=3):
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scale_exper = scaler.fit_transform(data)

        # Apply PCA to data
        pca = PCA(n_components=pca_n, svd_solver='full')
        return pca.fit_transform(scale_exper)

    # Calculate Mahalanobis dist functions
    def is_pos_def(A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False
    
    def cov_matrix(data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
                print("Error: Covariance Matrix is not positive definite!")

    def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
        diff = data - mean_distr
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))
        return md

    def MD_detectOutliers(dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        outliers = []
        for i in range(len(dist)):
            if dist[i] >= threshold:
                outliers.append(i)  # index of the outlier
        return np.array(outliers)

    def MD_threshold(dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        return threshold
        


    def full_process(experiment_n, components=2, chi2_print=True, exper_num=None):
        """Experiment data should only contain the columns that are desireable"""
        exper_pca = scale_decompose(experiment_n, pca_n=components)

        cov, inv_cov = cov_matrix(exper_pca)
        mean_dist = exper_pca.mean(axis=0)

        m_dist = MahalanobisDist(inv_cov, mean_dist, exper_pca)
    
        if chi2_print:
            fig_x = go.Figure(
            data=[go.Histogram(x=np.square(m_dist))],
            layout=go.Layout({'title': 'X^2 Distribution'})
            )
            fig_x.show()
    
        if exper_num:
            title = 'Mahalanobis Distribution Experiment {}'.format(exper_num)
        else:
            title = 'Mahalanobis Distribution'
        fig_m = pff.create_distplot([m_dist], group_labels=['m_dist'], bin_size=0.15)
        fig_m.update_layout(title_text=title)
        fig_m.show()
    
        return exper_pca, m_dist

    def corr_actualcommand(data, corr_cols):
        look_data = data[corr_cols]
        corr_data = look_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=list(corr_data.index),
            y=list(corr_data.columns)
        ))
        fig.show()

    # Unworn data
    # Dataset loaded above
    # Clean up data
    exper1_clean = clean_data(experiment1)
    clean_ex1 = exper1_clean[(exper1_clean['M1_CURRENT_FEEDRATE']!=50) & (exper1_clean['X1_ActualPosition']!=198)]
    print('')
    st.write("-------------------------------------------------------------------------")
    st.write('Length of data before inaccurate points removed {:d}'.format(len(exper1_clean)))
    st.write('Length of data after inaccurate points removed {:d}'.format(len(clean_ex1)))
    st.write("-------------------------------------------------------------------------")

    columns = exper1_clean.columns
    corr_cols = list(filter(lambda x: ('Actual' in x) | ('Command' in x), columns))
    corr_actualcommand(exper1_clean, corr_cols) 


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exper1_clean['index'], y=exper1_clean['X1_ActualPosition'],
        mode='lines',
        name='x-actual'
    ))
    fig.add_trace(go.Scatter(
        x=exper1_clean['index'], y=exper1_clean['Y1_ActualPosition'],
        mode='lines',
        name='y-actual'
    ))

    fig.show()


    # Worn data
    # Load data
    experiment6 = pd.read_csv(data_dir + 'experiment_06.csv')
    experiment6.reset_index(inplace=True)

    # Clean up data
    exper6_clean = clean_data(experiment6)
    clean_ex6 = exper6_clean[(exper6_clean['M1_CURRENT_FEEDRATE']!=50) & (exper6_clean['X1_ActualPosition']!=198)]
    st.write("--------------------------------------------------------------------------")
    st.write('Length of data before inaccurate points removed {:d}'.format(len(exper6_clean)))
    st.write('Length of data after inaccurate points removed {:d}'.format(len(clean_ex6)))
    st.write("--------------------------------------------------------------------------")

    columns = exper6_clean.columns
    corr_cols = list(filter(lambda x: ('Actual' in x) | ('Command' in x), columns))
    corr_actualcommand(exper6_clean, corr_cols)

    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(
        go.Scatter(
        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
        mode='lines',
        name='x-actual'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
        x=exper6_clean['index'], y=exper6_clean['Y1_ActualPosition'],
        mode='lines',
        name='y-actual'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
            mode='lines',
            name='x-actual'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['Z1_ActualPosition'],
            mode='lines',
            name='z-actual'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
            mode='lines',
            name='x-actual'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['S1_ActualVelocity'],
            mode='lines',
            name='s-actual'
        ),
        row=3, col=1
    )

    fig.show()

    columns = exper6_clean.columns
    raw_cols = list(filter(lambda x: ('Current' in x) | ('Voltage' in x) | ('Power' in x), columns))
    corr_actualcommand(exper6_clean, raw_cols+['X1_ActualPosition'])


    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
            mode='lines',
            name='x-actual'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['S1_OutputVoltage'],
            mode='lines',
            name='s-volt'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
            mode='lines',
            name='x-actual'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['S1_OutputPower']*100,
            mode='lines',
            name='s-power'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],
            mode='lines',
            name='x-actual'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=exper6_clean['index'], y=exper6_clean['S1_DCBusVoltage']*100,
            mode='lines',
            name='s-bus'
        ),
        row=3, col=1
    )

    fig.show()

    # Collect columns that are desireable in further analysis
    keeper_cols = list(filter(lambda x: 'Z1' not in x, raw_cols))

    # Define number of PCA components
    component = 2 # should use 2 because this fits the basic definition of the Mahalanobis distance

    # Perform outlier analysis
    exper1_pca, exper1_mdist = full_process(exper1_clean[keeper_cols], components=component)

    thresh = MD_threshold(exper1_mdist)

    exper6_pca, exper6_mdist = full_process(exper6_clean[keeper_cols], components=component)


    # Load another dataset - worn
    experiment8 = pd.read_csv(data_dir + 'experiment_08.csv')
    experiment8.reset_index(inplace=True)

    # Clean up data
    exper8_clean = clean_data(experiment8)
    clean_ex8 = exper8_clean[(exper8_clean['M1_CURRENT_FEEDRATE']!=50) & (exper8_clean['X1_ActualPosition']!=198)]
    print('')
    st.write("--------------------------------------------------------------------------")
    st.write('Length of data before inaccurate points removed {:d}'.format(len(exper8_clean)))
    st.write('Length of data after inaccurate points removed {:d}'.format(len(clean_ex8)))
    st.write("--------------------------------------------------------------------------")

    # Perform outlier analysis
    exper8_pca, exper8_mdist = full_process(exper8_clean[keeper_cols], components=component)


    # Load another dataset - unworn
    experiment3 = pd.read_csv(data_dir + 'experiment_03.csv')
    experiment3.reset_index(inplace=True)

    # Clean up data
    exper3_clean = clean_data(experiment3)
    clean_ex3 = exper3_clean[(exper3_clean['M1_CURRENT_FEEDRATE']!=50) & (exper3_clean['X1_ActualPosition']!=198)]
    print('')
    st.write("--------------------------------------------------------------------------")
    st.write('Length of data before inaccurate points removed {:d}'.format(len(exper3_clean)))
    st.write('Length of data after inaccurate points removed {:d}'.format(len(clean_ex3)))
    st.write("--------------------------------------------------------------------------")
    exper3_pca, exper3_mdist = full_process(exper3_clean[keeper_cols], components=component)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exper1_clean.reset_index().index, y=exper1_mdist,
        mode='lines',
        name='unworn_01'
    ))
    fig.add_trace(go.Scatter(
        x=exper6_clean.reset_index().index, y=exper6_mdist,
        mode='lines',
        name='worn_06'
    ))
    fig.add_trace(go.Scatter(
        x=exper8_clean.reset_index().index, y=exper8_mdist,
        mode='lines',
        name='worn_08'
    ))
    fig.add_trace(go.Scatter(
        x=exper3_clean.reset_index().index, y=exper3_mdist,
        mode='lines',
        name='unworn_03'
    ))
    fig.add_shape(
        type='line',
        y0=thresh,
        y1=thresh,
        x0=0,
        x1=max([len(exper1_mdist), len(exper6_mdist)]),
        line=dict(color='RoyalBlue', width=2, dash='dot')
    )
    fig.update_shapes(dict(xref='x', yref='y'))
    fig.show()

    completed_exper = outcomes[outcomes['machining_finalized']=='yes']

    unworn = []
    idx_unworn = []
    worn = []
    idx_worn = []
    for i, r in completed_exper.iterrows():
        if r['tool_condition'] == 'unworn':
            if r['No'] < 10:
                unw_data = pd.read_csv(data_dir + 'experiment_0{}.csv'.format(r['No']))
            else:
                unw_data = pd.read_csv(data_dir + 'experiment_{}.csv'.format(r['No']))
                unw_data['Experiment'] = r['No']
            unworn.append(unw_data)
            idx_unworn.append(r['No'])
        elif r['tool_condition'] == 'worn':
            if r['No'] < 10:
                w_data = pd.read_csv(data_dir + 'experiment_0{}.csv'.format(r['No']))
            else:
                w_data = pd.read_csv(data_dir + 'experiment_{}.csv'.format(r['No']))
            w_data['Experiment'] = r['No']
            worn.append(w_data)
            idx_worn.append(r['No'])
    
    unworn_df = pd.concat(unworn, ignore_index=True)
    worn_df = pd.concat(worn, ignore_index=True)

    unworn_clean = clean_data(unworn_df)
    worn_clean = clean_data(worn_df)

    reduce_unworn = unworn_clean[(unworn_clean['M1_CURRENT_FEEDRATE']!=50) & (unworn_clean['X1_ActualPosition']!=198)]
    reduce_worn = worn_clean[(worn_clean['M1_CURRENT_FEEDRATE']!=50) & (worn_clean['X1_ActualPosition']!=198)]

    #remove bad experiment 
    unworn_clean = unworn_clean[unworn_clean['Experiment']!=2]
    #perform outlier analysis
    unworn_pca, unworn_mdist = full_process(unworn_clean[keeper_cols], components=component)
    thresh = MD_threshold(unworn_mdist)
    print('Threshold: {:0.2f}'.format(thresh))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(0, len(unworn_mdist))),
        y=unworn_mdist,
        mode='lines',
        name='unworn'
    ))

    fig.add_shape(
        type='line',
        y0=thresh,
        y1=thresh,
        x0=0,
        x1=len(unworn_mdist),
        line=dict(color='RoyalBlue', width=2, dash='dot')
    )
    fig.update_shapes(dict(xref='x', yref='y'))
    fig.update_layout(title_text='Mahalanobis Distance Trance All Unworn Data')
    fig.show()


    unworn_test = pd.DataFrame(unworn_pca)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=unworn_test[0],
        y=unworn_test[1],
        mode='markers',
        name='unworn'
    ))

    fig.update_layout(title_text='PCA Plot')
    fig.show()

    # Perform outlier analysis
    worn_pca = dict()
    worn_mdist = dict()
    for ix in idx_worn:
        worn_pca_n, worn_mdist_n = full_process(
            worn_clean[worn_clean['Experiment']==ix][keeper_cols],
            components=component,
            chi2_print=False,
            exper_num=ix
        )
        worn_pca[ix] = worn_pca_n
        worn_mdist[ix] = worn_mdist_n   

    fig = go.Figure()

    x_size = []
    for key in worn_mdist:
        x_size.append(len(worn_mdist[key]))
        fig.add_trace(go.Scatter(
            x=list(range(0, len(worn_mdist[key]))),
            y=worn_mdist[key],
            mode='lines',
            name='exper_{}'.format(key)
        ))

    fig.add_shape(
        type='line',
        y0=thresh,
        y1=thresh,
        x0=0,
        x1=max(x_size),
        line=dict(color='RoyalBlue', width=2, dash='dot')
    )
    fig.update_shapes(dict(xref='x', yref='y'))
    fig.update_layout(title_text='Mahalanobis Distance Trance of Each Worn Experiment')
    fig.show()



if result == 'Tool Wear Detection':
    st.subheader('Tool Wear Detection')
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    frames = list()
    results = pd.read_csv("E:/hset/New folder/train.csv") 
    for i in range(1,19):
        exp = '0' + str(i) if i < 10 else str(i)
        frame = pd.read_csv("E:/hset/New folder/experiment_{}.csv".format(exp))
        row = results[results['No'] == i]
        frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
        frames.append(frame)
    df = pd.concat(frames, ignore_index = True)
    df.head()

    df_correlation=df.corr()
    df_correlation.dropna(thresh=1,inplace=True)
    df_correlation.drop(columns=['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia','target'],inplace=True)
    plt.figure(figsize=(20,20))
    sns.heatmap(df_correlation)
    st.pyplot(plt)
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import metrics


    import xgboost as xgb
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    import gc # for deleting unused variables
    

    #Creating Test Train Splits
    x=df.drop(columns=['target','Machining_Process'],axis=1)
    y=np.array(df['target'])
    X_train,X_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=100)
    
    #XgBoost

    xgb_model=XGBClassifier()
    xgb_model.fit(X_train,y_train)

    # make predictions for test data
    # use predict_proba since we need probabilities to compute auc
    y_pred = xgb_model.predict(X_test)
    y_pred[:10]

    st.write('-------------------------------------------------------------------------------------------------')
    st.write("Trained on {0} observations and scoring with {1} test samples.".format(len(X_train), len(X_test)))
    st.write('-------------------------------------------------------------------------------------------------')
    # roc_auc
    st.write('-------------------------------------------------------------------------------------------------')
    st.write("auc_score")
    #y_pred=y_pred.round()
    auc = roc_auc_score(y_test, y_pred)
    auc
    st.write('-------------------------------------------------------------------------------------------------')
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc


    def draw_roc( actual, probs ):
        fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                                drop_intermediate = False )
        auc_score = metrics.roc_auc_score( actual, probs )
        plt.figure(figsize=(6, 4))
        plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        return fpr, tpr, thresholds


    draw_roc(y_test,y_pred)


    # Error terms
    #Actual vs Predicted
    #c = [i for i in range(1,(len(y_test)+1),1)]
    count_points=70
    c = [i for i in range(1,count_points+1,1)]
    fig = plt.figure()
    plt.plot(c,y_test[:count_points], color="blue", linewidth=2.5, linestyle="-")#Actual Plot in blue
    plt.plot(c,y_pred[:count_points], color="red",  linewidth=2.5, linestyle="--")#predicted Plot in red
    fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('Worn_status', fontsize=16)    
    st.pyplot(fig) 


    
    st.write('-------------------------------------------------------------------------------------------------')

    st.subheader("Feature importances:-")
    features = [(df.columns[i], v) for i,v in enumerate(xgb_model.feature_importances_)]
    features.sort(key=lambda x: x[1], reverse = True)
    for item in features[:10]:
        st.write("{0}: {1:0.4f}".format(item[0], item[1]))
    st.write('-------------------------------------------------------------------------------------------------')


if result == 'Tips':
    st.subheader('Tips')
    st.markdown('''
    # Increase Tool Life, Reduce Tool Wear :-
    #### This guide will show you 11 ways to radically increase your tool life and reduce tool wear. Plus it will explain the details and mechanisms of tool wear, discuss how to calculate tool wear, and describe tool life monitoring.
    #### customer wanting to know how to maximize his tool life and reduce tool wear.  He’s doing long production runs and wants to keep the spindles turning as much as possible.  It was a good reminder that this is a topic on a lot of machinist’s minds so here are 11 tips to increase your tool life with lots of links to even more in-depth information in each area:

    ## 1.Use the Right Feeds and Speeds:
    #### You can go by how the cut sounds.
    #### You can’t possibly go wrong by just slowing things way down.
    #### You can just read the best feeds and speeds out of a handbook or tooling catalog.
    #### CNC feeds and speeds are just like manual feeds and speeds.

    ## 2.Keep Deflection Under Control :
    #### Deflection kills endmills, sometimes in surprising ways, and especially carbide endmills since they’re brittle and don’t bend as easily as HSS endmills.  Most machinists are unaware how much deflection they’re running until it gets too far out of hand.
    #### But, a good Feeds and Speeds Calculator will tell you how much deflection your cut parameters will generate.  A great one will help you optimize your cut parameters within deflection limits.  Also, when you’re setting up tools for use in many jobs, use as little Tool Stickout as possible.

    ## 3.Avoid Recutting Chips :
    #### Make sure the coolant is set up to get rid of them.  Sometimes flood coolant turns into “dribble” coolant because machines lack full enclosures and the machinist wants to avoid a mess.  Use mist for those machines as the dribble just covers up the chips sitting in the cut so you can no longer see them.

    ## 4.Lubricate Sticky Materials:
    #### Built up edge or “BUE” is the technical term.  Some materials have an affinity for what cutters are made of and they will weld chips onto the cutting edge which quickly results in a broken cutter.  Aluminum is one such, but there are a lot of others.  Look up the material and if it is prone to chip welding, you need lubrication.  You can get it from flood coolant, mist coolant, or some tool coatings.  What you can do is machine materials prone to chip welding with lubrication.

    ## 5.Add a Surface Speed Safety Factor:
    #### Given a choice between reducing surface speed (SFM or spindle rpms) and reducing chip load, surface speed is the one to go after for tool life unless you’re breaking relatively new cutters, in which case you need to reduce chip load.

    ## 6. Dial Back the Tortoise-Hare Slider :
    #### The Tortoise-Hare slider can be used to emphasize either Material Removal Rate (“Hare” end) or Surface Finish and Tool Life (“Tortoise” end).  Try dialing back more towards the Tortoise end when you’re particularly concerned about Tool Life.  What that will do is reduce both the Surface Speed and the chip loads, although it reduces chip load the most.

    ## 7.Use a Gentler Cut Entry in Your CAM Program:
    #### A lot of cutter wear starts on entry to the cut.  You may even chip the edge there, especially in tough work-hardening materials.  The solution to this problem is to adopt gentler entries.  Avoid plunging the cutter.  Instead, use one of these strategies:
    +##### Ramp in, with a relatively gentle ramp.
    +##### Helix or spiral in.
    +##### Use a decent-sized indexable drill to create a hole for entry.
    +##### For profile cuts and surfacing, arc into the cut.

    ## 8.Be Gentle Exiting the Cut Too
    #### The other reason to check out that toolpath article is that how you exit the cut matters too for tool life.

    ## 9.Rough With Tougher Tools
    #### Are you roughing with the same endmill you will use for finishing?  Same size and geometry anyway?
    #### There are better approaches.  Use a bigger tougher tool for the roughing.  Indexable endmills and corncob roughers can take a lot more abuse than solid endmills.

    ## 10. Spread Wear Over the Cutting Edge
    #### Are you keeping your cut depths super shallow thinking that’ll mean you’re taking it easy on the cutter?  Well, you are taking it easy, but unfortunately you’re also concentrating all the wear on the tip of the flutes.  They can only last so long that way.  What you need to do is spread that wear over as much of the flute as you can by increasing cut depths.  You’ll have to back off cut width as a tradeoff and you’ll have to watch out for excessive deflection, but once you have those two under control, you’ll get a lot more life out of your cutters.  This has another benefit in that it gives the flutes more “air cutting” time per revolution, which makes it easier for them to cool down as well as to get rid of chips. 

    ## 11.Minimize Runout
    #### Runout is a nasty business for cutters.  It’ll break tiny micromachining cutters in a heartbeat.  Larger cutters it just wears out prematurely.  Many tooling manufacturers estimate every tenth (0.0001″) of spindle runout reduces tool life by 10%.  That’s significant!''')



if result == 'Data Analysis':
    st.subheader('Data Analysis')
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import seaborn as sns
    sns.set(rc = {'figure.figsize':(30,16)})
    import os
    experiments = []
    for dirname, _, filenames in os.walk('E:/hset/New folder/'):
        for filename in filenames:
            if filename.startswith('experiment'):
                df = pd.read_csv(f"{dirname}/{filename}", index_col=None, header=0)
                df['Experiment'] = int(filename[-6:-4])
                experiments.append(df)
            
    series = pd.concat(experiments, axis=0, ignore_index=True)

    # add info if a step (point) is a milling one or a service one (preparation, repositioining etc.)
    series['Milling'] = series['Machining_Process'].str.startswith('Layer')


    # show tool trajectory - only for milling steps
    series[series.Milling].plot(x='X1_ActualPosition',y='Y1_ActualPosition');
    series = series[series.Experiment == 1]
    series['Velocity'] = np.sqrt(series['X1_ActualVelocity'] ** 2 + series['Y1_ActualVelocity'] ** 2+ series['Z1_ActualVelocity'] ** 2)
    series['Velocity'].plot() # all steps - including preparation etc.
    series[series.Milling]['Velocity'].plot()# only milling steps
    
    sns.scatterplot(data=series[series.Milling], x='X1_ActualPosition', y='Y1_ActualPosition', hue='Velocity', size='Velocity', sizes=(100, 300));
    sns.heatmap(series.corr());
    milling_pwr = series[series.Milling]['S1_OutputPower'].mean()
    not_milling_pwr = series[series.Milling == False]['S1_OutputPower'].mean()
    st.write('Milling pwr average:' + str(milling_pwr))
    st.write('Not Milling pwr average:' + str(not_milling_pwr))
    pct = milling_pwr / not_milling_pwr * 100
    pct
    sns.scatterplot(data=series, x='Machining_Process', y='S1_OutputPower');




