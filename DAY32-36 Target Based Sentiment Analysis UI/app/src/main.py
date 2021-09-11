from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
from inference import Predictor

app = Flask(__name__)
predictor = Predictor.build_prerequisites()


app.debug = True
@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/review_sentiment')
def review_sentiment():
    book_info = [1,2,3,4,45,5]
    return render_template('review_sentiment.html', book_info=book_info)


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    text = request.form['text']
    aspect = request.form['aspect']
    sentiment = predictor.execute(text, aspect)
    return jsonify({"sentiment": sentiment})


if __name__ == '__main__':
    app.run()
