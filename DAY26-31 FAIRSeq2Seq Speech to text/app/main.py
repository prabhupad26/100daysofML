from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from predict import Speech2Text

app = Flask(__name__)
app.debug = True


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/s2t_demo')
def s2t_demo():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio_data']
    file_name = secure_filename(file.filename)
    file_name = os.path.join("static", "uploads", f"{file_name}.webm")
    file.save(file_name)
    s2t = Speech2Text()
    return s2t.predict(file_name)


if __name__ == '__main__':
    app.run()
