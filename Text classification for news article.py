# %%
#CODE FOR TEXT classification for News article
#importing important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import string as s
import re
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes  import MultinomialNB 
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

# %%
#path to the dataset
train_file=r"C:\Users\prabh\Downloads\train_d.csv\train.csv"
test_file=r"C:\Users\prabh\Downloads\test_d.csv\test.csv"

# %%
#Reading te train data and test data
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)

# %%
#  Mapping Class Index: 1-World, 2-Sports, 3-Business, 4-Sci/Tech
train_data['Class'] = train_data['Class Index'].map({1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'})
test_data['Class'] = test_data['Class Index'].map({1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'})

# %%
train_data.head()

# %%
test_data.head()

# %%
#Exploring the dataset
print(f'train_data Shape:{train_data.shape}')
print(f'test_data Shape:{test_data.shape}')

# %%
train_data['Class'].value_counts()

# %%
test_data['Class'].value_counts()

# %%
#Countplot for train data
sns.countplot(x="Class",data=train_data)

# %%
#Count plot for test data
sns.countplot(x="Class",data=test_data)

# %%
# WordCloud for the Description based on Class Index it belong to
stop = set(stopwords.words('english'))
world = train_data[train_data['Class Index'] == 1]
world = world['Description']
sports = train_data[train_data['Class Index'] == 2]
sports = sports['Description']
business = train_data[train_data['Class Index'] == 3]
business = business['Description']
tech = train_data[train_data['Class Index'] == 4]
tech = tech['Description']
def wordcloud_draw(train_data, color = 'white'):
    words = ' '.join(train_data)
    cleaned_word = ' '.join([word for word in words.split()
    if (word != 'news' and word != 'text')])
    wordcloud = WordCloud(stopwords = stop, background_color = color,  width = 2500, height = 2500).generate(cleaned_word)
    plt.figure(1, figsize = (10,7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
print("world  related words:")
wordcloud_draw(world , 'white')
print("sports related words:")
wordcloud_draw(sports, 'white')
print("business related words:")
wordcloud_draw(business, 'white')
print("Tech related words:")
wordcloud_draw(tech, 'white')

# %%
#Merging the Title and Description together to form a new Column
train_data['news'] = train_data['Title'] +" " + train_data['Description']
test_data['news'] = test_data['Title'] +" " + test_data['Description']

# %% [markdown]
# Preprocessing of dataset

# %%
#Remove all tags 
def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)
train_data['news'] = train_data['news'].apply(remove_tags)
test_data['news'] = test_data['news'].apply(remove_tags)

# %%
# Remove Special Chars
def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews
train_data['news'] = train_data['news'].apply(special_char)
test_data['news'] = test_data['news'].apply(special_char)

# %%
#Remove Html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
train_data['news'] = train_data['news'].apply(remove_html)
test_data['news'] = test_data['news'].apply(remove_html)

# %%
#Remove urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
train_data['news'] = train_data['news'].apply(remove_urls)
test_data['news'] = test_data['news'].apply(remove_urls)

# %%
#Convert every car to lower case
def convert_lower(text):
    return text.lower()
train_data['news'] = train_data['news'].apply(convert_lower)
test_data['news'] = test_data['news'].apply(convert_lower)

# %%
def word_tokenize(text):
    tokens = re.findall("[\w']+", text)
    return tokens
train_data['news'] = train_data['news'].apply(word_tokenize)
test_data['news'] = test_data['news'].apply(word_tokenize)

# %%
#Remove stopwords
def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i.lower() not in stop:
            new_lst.append(i)
    return new_lst

train_data['news'] = train_data['news'].apply(remove_stopwords)
test_data['news'] = test_data['news'].apply(remove_stopwords)

# %% [markdown]
# Stemming of the dataset

# %%
def stemming(text):
    porter_stemmer = nltk.PorterStemmer()
    roots = [porter_stemmer.stem(each) for each in text]
    return (roots)

train_data['news'] = train_data['news'].apply(stemming)
test_data['news'] = test_data['news'].apply(stemming)

# %%
# Lemmatization of the Dataset
lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst

train_data['news'] = train_data['news'].apply(lemmatzation)
test_data['news'] = test_data['news'].apply(lemmatzation)

# %%
#Remove Extra Words
def remove_extrawords(lst):
    stop=['href','lt','gt','ii','iii','ie','quot','com']
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

train_data['news'] = train_data['news'].apply(remove_extrawords)
train_data['news'] = train_data['news'].apply(remove_extrawords)

# %%
train_data_X=train_data['news']
train_data_y=train_data['Class Index']
test_data_X=test_data['news']
test_data_y=test_data['Class Index']

# %%
train_data_X=train_data_X.apply(lambda x: ''.join(i+' ' for i in x))
test_data_X=test_data_X.apply(lambda x: ''.join(i+' '  for i in x))

# %%
n=20000 # Not enough memory so just took first n pre-processed texts

# %%
# used tfidf but could have used bag of words and other techniques
tfidf=TfidfVectorizer(min_df=8,ngram_range=(1,3))
train_1=tfidf.fit_transform(train_data_X[:n])
test_1=tfidf.transform(test_data_X[:n])

train_arr=train_1.toarray()
test_arr=test_1.toarray()

# %%
#training on Multinomial
NB_MN=MultinomialNB(alpha=0.52)
NB_MN.fit(train_arr,train_data_y[:n])
pred=NB_MN.predict(test_arr)

# %%
print("first 20 actual labels")
print(test_data_y.tolist()[:40])
print("first 20 predicted labels")
print(pred.tolist()[:40])

# %%
from sklearn.metrics  import f1_score,accuracy_score
print("F1 score of the model")
print(f1_score(test_data_y[:n],pred,average='micro'))
print("Accuracy of the model")
print(accuracy_score(test_data_y[:n],pred))
print("Accuracy of the model in percentage")
print(round(accuracy_score(test_data_y[:n],pred)*100,3),"%")

# %%
#Confusion Matrix
from sklearn.metrics import  confusion_matrix
sns.set(font_scale=1.5)
cof=confusion_matrix(test_data_y[:n], pred)
cof=pd.DataFrame(cof, index=[i for i in range(1,5)], columns=[i for i in range(1,5)])
plt.figure(figsize=(8,8))

sns.heatmap(cof, cmap="PuRd",linewidths=1, annot=True,square=True,cbar=False,fmt='d',xticklabels=['World','Sports','Business','Science'],yticklabels=['World','Sports','Business','Science'])
plt.xlabel("Predicted Class");
plt.ylabel("Actual Class");

plt.title("Confusion Matrix for News Article Classification");

# %%
# Create list of model and accuracy dicts
perform_list = [ ]

# %%
#trying different models for training and picking up the best based on accuracy, f1 score
def run_model(model_name, est_c, est_pnlty):
    mdl = ""
    if model_name == 'Logistic Regression':
        mdl = LogisticRegression()
    elif model_name == 'Random Forest':
        mdl = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)
    elif model_name == 'Support Vector Classifer':
        mdl = SVC()
    elif model_name == 'Decision Tree Classifier':
        mdl = DecisionTreeClassifier()
    elif model_name == 'K Nearest Neighbour':
        mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
    elif model_name == 'Gaussian Naive Bayes':
        mdl = GaussianNB()
    oneVsRest = OneVsRestClassifier(mdl)
    oneVsRest.fit(train_arr, train_data_y[:n])
    y_pred = oneVsRest.predict(test_arr)
    # Performance metrics
    accuracy = round(accuracy_score(test_data_y[:n], y_pred) * 100, 2)
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(test_data_y[:n], y_pred, average='micro')
    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'F1-score : {f1score}')
    # Add performance parameters to list
    perform_list.append(dict([('Model', model_name),('Test Accuracy', round(accuracy, 2)),('Precision', round(precision, 2)),('Recall', round(recall, 2)),('F1', round(f1score, 2))]))

# %%
run_model('Logistic Regression', est_c=None, est_pnlty=None)

# %%
run_model('Random Forest', est_c=None, est_pnlty=None)

# %%
run_model('Support Vector Classifer', est_c=None, est_pnlty=None)

# %%
run_model('Decision Tree Classifier', est_c=None, est_pnlty=None)

# %%
run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)

# %%
run_model('Gaussian Naive Bayes', est_c=None, est_pnlty=None)



