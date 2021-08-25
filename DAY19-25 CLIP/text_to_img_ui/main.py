from flask import Flask, flash, redirect, render_template, request, session, abort, json
from time import strftime
import base64
import json
from clip_inference import ClipInference

app = Flask(__name__)

app.debug = True


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/clip_demo')
def clip_demo():
    return render_template('clip_demo.html')


@app.route('/get_image_results', methods=['POST'])
def get_image_results():
    query = request.json["query"]
    model_inference = ClipInference()
    probability, img_feat = model_inference.image_from_text(query)
    model_inference.convert_image(img_feat)
    file_name = model_inference.save_image(query, probability, img_feat)
    probability = str(probability.squeeze().cpu().numpy().astype(str))
    return {"probability": probability, "file_name": file_name}, 200


@app.route('/get_text_results', methods=['POST'])
def get_text_results():
    image_name = request.files["file"]
    file_name = request.form['filename']
    image_name.save(f"static/{file_name}")
    model_inference = ClipInference()
    file_name = model_inference.text_from_image(file_name)
    return {"file_to_be_displayed": file_name}, 200


if __name__ == '__main__':
    app.run()
