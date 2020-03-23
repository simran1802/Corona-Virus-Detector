from flask import Flask, render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()
@app.route('/', methods=["GET","POST"])
def hello():
    if request.method == "POST":
        myDict = request.form
        
        fever = int(myDict['fever'])
        body_pain = int(myDict['pain'])
        age = int(myDict['age'])
        runny_nose = int(myDict['nose'])
        diff_breath = int(myDict['breath'])
        
        # input = [102,1,1,0]
        input = [fever,body_pain,runny_nose,diff_breath]
        info_prob = clf.predict_proba([input])[0][1]
        # print(info_prob)
        # return 'hello' + str(info_prob)
        return render_template('show.html',inf=round(info_prob*100))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    