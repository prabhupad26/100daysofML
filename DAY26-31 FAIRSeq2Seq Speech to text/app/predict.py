import os.path
import subprocess
import librosa
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
import torch


class Speech2Text:
    def __init__(self):
        # self.model, self.processor = self.load_model()
        pass

    def read_file(self, file):
        print("Reading sound file")
        file = os.path.join('static', 'uploads', file)
        file = self.convert_to_wav(file)
        speech_arr, sampling_rate = librosa.load(file, sr=16000)
        return speech_arr, sampling_rate

    def convert_to_wav(self, file):
        print("Converting to wav file")
        new_file = f'{file[:-4]}wav'
        command = ['ffmpeg', '-i', file, '-sample_rate', '16000', new_file]
        subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        return new_file

    def load_model(self):
        print("Loading the model")
        processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")
        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")
        return model, processor

    def predict(self, file_name):
        print("Stating prediction")
        file_name = os.path.basename(file_name)
        speech_arr, sampling_rate = self.read_file(file_name)
        model, processor = self.load_model()
        input_values = processor(speech_arr, sampling_rate=sampling_rate,
                                 return_tensors="pt").input_values
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.decode(pred_ids[0])


if __name__ == '__main__':
    file_name = "2021-09-22T063239.237Z.webm"
    s2t = Speech2Text()
    print(s2t.predict(file_name))
