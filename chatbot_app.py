from flask import Flask, request, render_template,session,jsonify,Response
import pickle
import warnings
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import re
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)




app = Flask(__name__)
app.secret_key = 'rohit_biradar'
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

#clf = pickle.load(open('dt.pkl', 'rb'))
#le = pickle.load(open('le.pkl', 'rb'))
training = pd.read_csv('train.csv')
y = training['prognosis']
le = preprocessing.LabelEncoder()
le= le.fit(y)

cols= training.columns
cols= cols[:-1]

x = training[cols]
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)

reduced_data = training.groupby(training['prognosis']).max()

def SeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
def Description():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def precautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def calc_severity(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        return "You should take the consultation from doctor. "
    else:
        return "It might not be that bad but you should take precautions."

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]           
def second_predict(symptoms_exp):
    df = pd.read_csv('train.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))
         

def dicision_tree_code(tree, feature_names,msg):


    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = msg
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=second_predict(symptoms_exp)
            # print(second_prediction)
            calc_severity(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)

def disease_search(feature_names,msg):
    
    return_string=''
    res=''   
    ret=''
    chk_dis=",".join(feature_names).split(",")
    conf,cnf_dis=check_pattern(chk_dis,msg)
    if conf==1:
        session['searches']=cnf_dis
        for num,it in enumerate(cnf_dis):
            return_string=return_string+str(num)+') '+it+'<br/>'
            

        if num!="":
            res='Searches related to the input:'+'<br/>'+return_string+'<br/>'+"Select the one you meant (0 -{}) : ".format (num) 
            session['confirm'] =1
        else:
            res='No related data found.Please try again with another input'
            session['again'] = 1

    else:      
        res='Please enter valid symptoms'
        session['again'] = 1   
    return res

def chatbot_response(tree,feature_names,msg):  
    SeverityDict()
    Description()
    precautionDict() 
    c=""   
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []
    pre=""
    num_days = session.get('num_days', 0)
    inp= session.get('inp', 0)
    flag_rec = session.get('flag_rec', 0)
    disease_input = session.get('disease_input', 0)
    syms_exp=session.get('syms_exp', 0)
    second_prediction = ""
    def recurse(node, depth):
        c=""
        pre=""
        second_prediction = ""
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            sym_list=[]
            for syms in list(symptoms_given):
                sym_list.append(syms)
            l=len(sym_list)
            print (f" symptoms are: {sym_list}") 
            print (f" length of list: {l}")    

            session['syms']=1
            syms = session.get('syms', 0)
            count=session.get('count', 0)
            
            
            if count < len(sym_list):
                if count >=1 and count< len(sym_list):
                    while True:
                        if msg=="yes" or msg=="no":
                            break
                        else:
                            return "Provide proper answer i.e. (yes/no) : "
                    if(msg=="yes"):
                        
                        syms_exp.append(sym_list[count-1])
                        session['syms_exp']=syms_exp
                            
                        print(f"Exp symptoms are: {syms_exp}")
                count=count+1
                print (f"count is: {count}")
                session['count']=count
                if count==len(sym_list):
                    session['syms']=0
                return "Are you experiencing any {}".format(sym_list[count-1] +"?")   
            else:
                second_prediction=second_predict(syms_exp)
                c=calc_severity(syms_exp,num_days)
                if(present_disease[0]==second_prediction[0]):
                    precution_list=precautionDictionary[present_disease[0]]
                    for  i,j in enumerate(precution_list):
                            pre=pre+str(i+1)+') '+j+'<br/>'
                    return "Severity: "+'<br/>'+c+'<br/>'+'<br/>'+"Predicted disease: "+'<br/>'+"You may have {}" .format(present_disease[0])+'<br/>'+ description_list[present_disease[0]]+'<br/>'+'<br/>'+"precaution"+'<br/>'+'Take the following measures:'+'<br/>'+pre+'<br/>'
                            
                else:
                    precution_list=precautionDictionary[present_disease[0]]
                    for  i,j in enumerate(precution_list):
                            pre=pre+str(i+1)+') '+j+'<br/>'
                    return "Severity: "+'<br/>'+c+'<br/>'+'<br/>'+"Predicted disease: "+'<br/>'+"You may have {}" .format(present_disease[0])+ " or "+ "{}" .format(second_prediction[0])+'<br/>'+'<br/>'+ description_list[present_disease[0]]+'<br/>'+description_list[second_prediction[0]]+'<br/>'+'<br/>'+"Precaution: "+'<br/>'+'Take the following measures:'+'<br/>'+pre
    return recurse(0,1)

@app.route('/')
def home():
    session['num_responses'] = 0
    session['confirm'] = 0
    session['searches'] = ''
    session['disease'] = 0
    session['num_days'] = 0
    session['hi'] = 0
    session['flag_rec'] = 0
    session['disease_input'] = ""
    session['syms_exp']=[]
    session['syms']=0
    session['count']=0
    session['again'] = 0
    return render_template('index.html')


@app.route("/get")
def get_bot_response():
    
    userText1 = request.args.get('msg')
    session['num_responses'] = session.get('num_responses', 0) + 1
    num_responses = session.get('num_responses', 0)
    num_days=session.get('num_days', 0)
    confirm=session.get('confirm', 0)
    hi=session.get('hi', 0)
    disease=session.get('disease', 0)
    again=session.get('again', 0)
    flag_rec = session.get('flag_rec', 0)
    if num_responses==1:
        session['hi'] = 1
        return "Hello {}!" .format(userText1)+'<br/>'+'<br/>'+"Enter the symptom you are experiencing:"
        
    elif confirm==1:
        searches=session.get('searches', 0)
        disease_in=searches[(int(userText1))]
        session['disease_input'] = disease_in
        session['confirm'] = 0
        session['disease'] = 1   
        return "So you are experiencing {}".format(disease_in)+'<br/>'+"Hmmm Okay! From how many days?"
    elif disease==1:
        try:
            n_days=int(userText1)
            session['num_days'] = n_days 
            session['disease'] = 0
            bot_response = chatbot_response(clf,cols,userText1)
            print(f"bot_response: {bot_response}")
            return bot_response
        except:
            return"Enter valid number of days."
    elif hi==1 or again==1:
        session['hi'] = 0
        return disease_search(cols,userText1)
    
    
    bot_response = chatbot_response(clf,cols,userText1)
    print(f"bot_response: {bot_response}")
    return bot_response
if __name__ == "__main__":
    app.run(debug=True)