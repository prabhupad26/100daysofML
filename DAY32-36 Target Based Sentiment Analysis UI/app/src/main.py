from flask import Flask, render_template, request
from inference import Predictor
from utils.fetch_data import fetch_reviews_data

app = Flask(__name__)
predictor = Predictor.build_prerequisites()
init_data, product_dict = fetch_reviews_data()

app.debug = True


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/review_sentiment')
def review_sentiment():
    return render_template('review_sentiment.html', product_data=product_dict, reviews=init_data)


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    aspect = request.form['aspect']
    data_ = init_data.copy()
    modified_data = predictor.execute(data_, aspect)
    return render_template('review_sentiment.html', product_data=product_dict, reviews=modified_data)


if __name__ == '__main__':
    app.run()
