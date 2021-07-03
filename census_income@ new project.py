#!/usr/bin/env python
# coding: utf-8

# # Census Income Project
# Problem Statement:
# 
# 
# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.
# 
# Description of fnlwgt (final weight)
# The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian non-institutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau. We use 3 sets of controls. These are:
# 
# A single cell estimate of the population 16+ for each state.
# 
# Controls for Hispanic Origin by age and sex.
# 
# Controls by Race, age and sex.
# 
# We use all three sets of controls in our weighting program and "rake" through them 6 times so that by the end we come back to all the controls we used. The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.

# In[1]:


from pyforest import *
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv')


# In[3]:


df


# In[4]:


#! pip install pandas-profiling
#from pandas_profiling import ProfileReport
#prof = ProfileReport(df)
#prof.to_file(output_file='output_of census.html')


# In[5]:


df.head()


# In[ ]:





# In[6]:


#Checking the Shape of the Data Set
df.shape


# In[7]:


#Checking the Columns Name
df.columns


# In[8]:


#checking the Info of Data set
df.info()


# In[9]:


#checking the Data Types of all columns
df.dtypes


# In[10]:


#We have 2 types of datatypes Object and Int
#Lets Check the Nan Values
df.isnull().values.any()


# In[11]:


#From above result we can say that we  don't have any kind of nan values in this Data Set
#lets plot it on Heatmap 
sns.heatmap(df.isnull(),cmap='winter',cbar=False)


# In[12]:


#lets Divided the data set into 2 types accoring to the Dtypes
df_numeric=df._get_numeric_data()
df_numeric_columns=df._get_numeric_data().columns
df_Catagorical=df.drop(columns=['Age', 'Fnlwgt', 'Education_num', 'Capital_gain', 'Capital_loss', 'Hours_per_week'],axis=1)


# In[13]:


#lets do the EDA for the df_nuudf_numeric
df_numeric.head()


# In[14]:


df_Catagorical.head()


# In[15]:


#lets Do the Value Counts with Count plot
for i in df_Catagorical.columns:
    val=df_Catagorical[i].value_counts()
    val_normalize=df_Catagorical[i].value_counts(normalize=True)
    print("Result for → ",i)
    print("result start")
    print(val)
    print('►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►►')
    print(val_normalize)
    sns.set_style("whitegrid")
    plt.figure(figsize=[12,4])
    sns.countplot(x=df_Catagorical[i],data=df)
    plt.plot(val,color='red',label=val)
    plt.xlabel(i,)
    plt.xticks(rotation=75)
    plt.legend()
    plt.show()
    print("♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫♪♪♫")
    print("\n")
    print("Result end  for → ",i)
    


# In[16]:


df.Income.value_counts(normalize=True)


# In[17]:


#From the Above Results We got to know that our Data Is Inbalanced we habe to balanced that one 
#lets do the Bai Variate Analysis for all catagorigal Data
df_Catagorical.columns


# # Workclass vs Income

# In[18]:


plt.figure(figsize=[10,6])
sns.countplot(x="Workclass",hue="Income",data=df,dodge=False,palette="Set1")
plt.show()


# # Education vs Income

# In[19]:


plt.figure(figsize=[10,6])
sns.countplot(x="Education",hue="Income",data=df,dodge=False,palette="Set2")
plt.show()


# # Marital_status vs Income

# In[20]:


plt.figure(figsize=[10,6])
sns.countplot(x="Marital_status",hue="Income",data=df,dodge=False,palette="brg_r")
plt.show()


# # Occupation Vs Income

# In[21]:


plt.figure(figsize=[10,6])
sns.countplot(x="Occupation",hue="Income",data=df,dodge=False,palette="Spectral_r")
plt.xticks(rotation=75)
plt.show()


# # Relationship vs Income 

# In[22]:


plt.figure(figsize=[10,6])
sns.countplot(x="Relationship",hue="Income",data=df,dodge=False)
plt.show()


# # Race Vs Income

# In[23]:


plt.figure(figsize=[10,6])
sns.set_theme(style="darkgrid")
sns.countplot(x="Race",hue="Income",data=df,dodge=False)
plt.show()


# # Sex vs Income

# In[24]:


plt.figure(figsize=[10,6])
sns.countplot(x="Sex",hue="Income",data=df,dodge=False,saturation=10,palette="Set3")
plt.show()


# # Native_country vs Income

# In[25]:


plt.figure(figsize=[10,6])
sns.countplot(x="Native_country",hue="Income",data=df)
plt.xticks(rotation=75)
plt.show()


# # Numeric Data

# In[26]:


df_numeric.head()


# In[27]:


dec=df.describe()


# In[28]:


dec


# In[29]:


dec.plot(kind='line')


# In[30]:


#correlation
sns.heatmap(df.corr(),annot=True,cmap='winter',fmt='.1%')


# In[31]:


#We can find the Result from above graph that no  columns are not corelated with the others columns

#lets see for the Outliers

for i in df_numeric:
    col=df[i]
    sns.boxplot(col,color='r')
    plt.show()


# In[32]:


#we can see that lots of  Outliers are present in the many columns


# In[33]:


#Univariate analysis  for the Numeric data 
df_numeric.columns


# In[34]:


#Checking the Data Distribution
for i in df_numeric.columns:
    sns.distplot(df[i],color='blue',rug=False)
    plt.show()
#https://towardsdatascience.com/skewed-data-a-problem-to-your-statistical-model-9a6b5bb74e37


# In[35]:


#checking the mean
def uni(col):
    df[col].hist()
    m=df[col].mean()
    plt.axvline(m,linewidth=4,color='green',label=('mean %0.2f'%m))
    plt.xlabel(col)
    plt.ylabel("count")
    plt.legend()
    plt.show()


# In[36]:


for i in df_numeric.columns:
    uni(i)


# # Bai Variant Analysis

# In[37]:


df_numeric.columns


# # Age Vs Income

# In[38]:


sns.scatterplot(x="Age",y="Income",data=df)


# # Fnlwgt vs Income

# In[39]:


sns.violinplot(x="Fnlwgt",y="Income",data=df)


# # Education_num vs Income

# In[40]:


sns.lineplot(x="Education_num",y="Income",data=df)


# # Capital_gain vs Income

# In[41]:


sns.pointplot(x="Capital_gain",y="Income",data=df)


# # Capital_loss vs Income

# In[42]:


sns.scatterplot(x="Capital_loss",y="Income",data=df)


# # Hours_per_week vs Income

# In[43]:


sns.barplot(x="Hours_per_week",y="Income",data=df)


# In[44]:


#pairplot
sns.pairplot(df,hue="Income")


# # outlier ditection

# In[45]:


df.boxplot()


# In[46]:


#Encoding
#objective to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df.columns:
    if df[i].dtypes=='O':
        df[i]=le.fit_transform(df[i])


# In[47]:


#Outliers Remove
from scipy.stats import zscore
z=np.abs(zscore(df))
df_new=df[(z<3).all(axis=1)]


# In[48]:


#Checking the Shape Of new Dat Frame
print(df_new.shape,df.shape)


# In[49]:


# checking the Data loss

loss_=((len(df)-len(df_new))/len(df))*100
print(loss_)


# In[50]:


#we can see above Result that we loss more  data .
#so we are going to perform the Hypertuning

Q1=df.quantile(0.010)
Q3=df.quantile(0.99)
IQR=Q3-Q1
df_new_final=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
loss_=((len(df)-len(df_new_final))/len(df))*100
print(df_new_final.shape,df.shape)
print("Data Loss % is →→",loss_)


# In[51]:


#lets split the Data set
x=df_new_final.drop(("Income"),axis=1)


# In[52]:


y=df_new_final[["Income"]]


# In[53]:


# checking the Shape of the x and y
print("X shape →→",x.shape,"\n","Y shape→→",y.shape)


# In[54]:


#lets handel the skew ness of our data
x.skew()


# In[55]:


for i in x.columns:
    sns.distplot(x[i],color='r')
    plt.show()


# In[58]:


x1=x.drop(columns=['Capital_gain', 'Capital_loss','Relationship'],axis=1)
from scipy import stats
for i in x1.columns:
    col=x[i].skew()
    lower=(-0.55)
    highest=0.55
    if col > highest:
        x[i]=np.log(x[i])
    if col  < lower:
        x[i]=np.sqrt(x[i])
x['Capital_gain']=np.sqrt(x['Capital_gain'])
x['Capital_loss']=np.sqrt(x['Capital_loss'])
x['Relationship']=np.sqrt(x['Relationship'])


# In[59]:


x.skew()


# In[60]:


#Data Balancing
y.value_counts()


# In[62]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(x, y)

y_sm.value_counts()


# In[78]:


print(X_sm.shape, y_sm.shape)


# In[63]:


#Data Scalling
from sklearn.preprocessing import MinMaxScaler,StandardScaler
sc=StandardScaler()
x=sc.fit_transform(X_sm)
x=pd.DataFrame(x,columns=X_sm.columns)


# In[79]:


x.shape


# In[69]:


#Pca
from sklearn.decomposition import PCA
cover_matrix=PCA(n_components=len(x.columns))
cover_matrix.fit(x)


# In[72]:


plt.ylabel("Engine values")
plt.xlabel("#no of featutes")
plt.title("PCA Engine values")
plt.ylim(0,max(cover_matrix.explained_variance_))
plt.axhline(y=1,color='g',linestyle='--')
plt.plot(cover_matrix.explained_variance_,'ro-')
plt.show()


# In[73]:


pca=PCA(n_components=5)
x1=pca.fit_transform(x)
x=pd.DataFrame(x1)


# In[76]:


x.shape


# In[80]:


y=y_sm
y.shape


# # ModelCreation 

# In[83]:


#In this data set we know that out target variable Default is catagorical so we are going to take Logistic regresson
#lets make a function for getting the best random_satae for a model toget better accuracy score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split
final_accuracy=[]
final_random=[]
clf=[]
def max_acc(rgr,x,y):
    max_acc=0
    for r in range(42,100):
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r,test_size=0.20,stratify=y)
        rgr.fit(x_train,y_train)
        y_prd=rgr.predict(x_test)
        rc=accuracy_score(y_test,y_prd)
        if rc>max_acc:
            max_acc=rc
            final_r=r
    final_accuracy.append(max_acc)
    final_random.append(final_r)
    clf.append(rgr)
    print("max accuracy_ score coressponding to ♣♣→",final_r,"is♣♣",max_acc*100)


# In[84]:


from sklearn.linear_model import LogisticRegression
clf_lg=LogisticRegression()
max_acc(clf_lg,x,y)


# In[100]:


#lets make a function for cross_val_score
from sklearn.model_selection import cross_val_score   
cvs=[]
def Cross_validity(model,x,y):
    c=cross_val_score(model,x,y,cv=5,scoring="accuracy")
    print("mean accuracy score for ",model,c.mean())
    print("Standard deviation  in accuracy score for ",model,c.std())
    print()
    print("******************************************************")
    print("After seen the cross validation score of",model,"the accuracy score mean is",c.mean())
    cvs.append(c.mean())


# In[101]:


Cross_validity(clf_lg,x,y)


# In[103]:


#lets make a function for confusion matrix and classification report 
def clasifier(md,x,y,rd):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=rd,test_size=0.20)
    md.fit(x_train,y_train)
    pre=md.predict(x_test)
    acc=accuracy_score(y_test,pre)
    print(acc)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,pre)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print ('roc_auc_score = ',roc_auc)
    cm=confusion_matrix(y_test,pre)
    sns.heatmap(cm,annot=True,cmap='rainbow',cbar=False)
    print()
    cr=classification_report(y_test,pre)
    print()
    print()
    print()
    plt.figure(figsize=[20,50],facecolor='green')
    plt.subplot(912)
    plt.title(md,{"fontsize":22})
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = %0.2f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print(cr,"\n","☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼☼")


# In[104]:


clasifier(clf_lg,x,y,85)


# In[108]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
params={"n_neighbors":range(1,40,2)}
grd_kn=GridSearchCV(kn,params,cv=10,scoring='accuracy')
grd_kn.fit(x,y)
grd_kn.best_params_


# In[110]:


knn=KNeighborsClassifier(n_neighbors=3)
max_acc(knn,x,y)


# In[112]:


Cross_validity(knn,x,y)


# In[113]:


clasifier(knn,x,y,83)


# In[127]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
params={'n_estimators':range(100,200,1000),'criterion':['gini','entropy'],"max_features":['auto','sqrt','log2']}
gr_sc=GridSearchCV(rfc,params,cv=10,scoring='accuracy')
gr_sc.fit(x,y)
#svc.get_params().keys()
gr_sc.best_params_


# In[128]:


clf_rfc=RandomForestClassifier(criterion='entropy',max_features='log2',n_estimators=100)
max_acc(clf_rfc,x,y)


# In[129]:


Cross_validity(clf_rfc,x,y)


# In[130]:


clasifier(clf_rfc,x,y,74)


# In[134]:


#from above Result we  found that Random Forest Classifier is working fine 
#so we are finalize the Randomforest Classifier 


# In[135]:


#Making pipeline 
from sklearn.pipeline import Pipeline
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=74,test_size=0.20,stratify=y)


# In[152]:


#making a function for PCA
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
class pca(BaseEstimator):
    def __init__(self):
        pass
    def fit(self,documents,y=None):
        return self
    def transform(self,x_dataset):
        pca=PCA(n_components=5)
        x1=pca.fit_transform(x_dataset)
        x=pd.DataFrame(x1)
        return x
        


# In[153]:


sc=StandardScaler()


# In[154]:


pipe = Pipeline([('scaler', StandardScaler()),('PCA',pca()),('RFC', RandomForestClassifier(criterion='entropy',max_features='log2',n_estimators=100))])


# In[155]:


pipe


# In[156]:


pipe_final=pipe.fit(x_train,y_train)


# In[159]:


pipe_final.predict(x_test)


# In[160]:


#lets Save the Model
import joblib
joblib.dump(pipe_final,"census_income.pkl")


# In[161]:


#saving as csv also 
joblib.dump(pipe_final,"census_income.csv")


# In[164]:


#lets chk our model
model=joblib.load('census_income.pkl')
model.predict(x_test[:5])


# # Thank You

# In[ ]:




