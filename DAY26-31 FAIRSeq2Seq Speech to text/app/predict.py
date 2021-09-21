import soundfile as sf
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
import torch


class Speech2Text:
    def __init__(self):
        self.model, self.processor = self.load_model()

    def read_file(self, file):
        speech_arr, sampling_rate = sf.read(file)
        return speech_arr, sampling_rate

    def load_model(self):
        processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")
        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")
        return model, processor

    def predict(self, file_name):
        speech_arr, sampling_rate = self.read_file(file_name)
        model, processor = self.load_model()
        input_values =  processor(speech_arr, sampling_rate=sampling_rate,
                                  return_tensors="pt").input_values
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.decode(pred_ids[0])


if __name__ == '__main__':
    file_name = "static/uploads/2021 - 09 - 21T165057.847Z.wav"
    s2t = Speech2Text()
    print(s2t.predict(file_name))
