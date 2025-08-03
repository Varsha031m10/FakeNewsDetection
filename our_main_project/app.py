from flask import Flask,render_template,url_for,request
import joblib
import re
import string
import pandas as pd


app = Flask(__name__)
Model = joblib.load(r"C:\Users\heman\Downloads\fake news detection\our_main_project\model22.pkl")

@app.route('/')
def index():
    return render_template("index.html")

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/prediction',methods=['POST','GET'])
def pre():
    if request.method == 'POST':
        news = request.form['news']
        news = wordpre(news)
        news = pd.Series(news)
        print(news)
        result = Model.predict(news)[0]
        if result==0:
            result="FAKE"
        else:
            result="REAL"
        return render_template("prediction.html",predection_text = "NEWS IS ",result=result)
    else:
        return render_template('prediction.html')
     
    

if __name__ == "__main__":
    app.run(debug=True)