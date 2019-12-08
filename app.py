from flask import Flask, render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("H_Clinton-emails_subset.csv")
    df_data = df[['RawText','MetadataSubject']]
    ##Features and Labels
    df_x = df_data['RawText']
    df_y = df_data.MetadataSubject
    #extract features with countvectorizer
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(cv)
    X_train, X_test, y_train, y_test = train_test_split(X,df_y, test_size=0.3, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)


    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction = my_prediction)





if __name__ == '__main__':
    app.run(debug=True)
