from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))
label_preprocessor = pickle.load(open('models/label_preprocessor.pkl', 'rb'))

@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    household = int(request.form['households'])
    ocean_proximity = request.form['ocean_proximity']
    ocean_proximity = int(label_preprocessor.transform([ocean_proximity])[0])
    data = [household, ocean_proximity]
    prediction = model.predict([data])

    print('Household', household)
    print('\ocean prox', ocean_proximity)
    print(data)
    print(prediction)
    #predictionString = ','.join(str(x) for x in prediction)
    output = round(prediction[0],2)
    return render_template('index.html', output = output)

if __name__ == '__main__':
    app.run(debug=True)