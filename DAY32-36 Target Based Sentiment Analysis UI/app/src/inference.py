import torch
from models.models import LSTM, AE_LSTM
from config.get_config import get_dataset_config, get_model_config
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer, Vocab
import os


class Predictor:
    def __init__(self, embed_matrix, tokenizer, model_name):
        self.model_name = model_name
        self.model_dict = {'AE': AE_LSTM, 'LSTM': LSTM}
        self.polarity_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.embed_matrix = embed_matrix
        self.tokenizer = tokenizer

    @classmethod
    def build_prerequisites(cls, model_name='AE', dataset_name='laptop_reviews'):
        files_ = get_dataset_config(dataset_name)
        tokenizer = build_tokenizer(
            fnames=[files_['file_train'], files_['file_test']],
            max_length=80,
            data_file=files_['file_tokenized'])
        embedding_matrix = build_embedding_matrix(
            vocab=tokenizer.vocab,
            embed_dim=200,
            data_file=files_['file_embed_matrix'])
        return cls(embedding_matrix, tokenizer, model_name)

    def build_model(self):
        checkpoint, dim = self.get_checkpoint()
        model = self.model_dict[self.model_name]
        model = model(self.embed_matrix, dim["embed_dim"],
                      dim["hidden_dim"], dim["polarities_dim"])
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def get_checkpoint(self):
        checkpoint_file, dim = get_model_config(self.model_name)
        print(f"Using checkpoint file : {checkpoint_file}")
        checkpoint_file = os.path.join(os.getcwd(), 'models', 'checkpoints', checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        return checkpoint, dim

    def execute(self, text, aspect=None):
        model = self.build_model()
        data = torch.tensor(self.tokenizer.text_to_sequence(text)).reshape(1, -1)
        if aspect:
            aspect = torch.tensor(self.tokenizer.text_to_sequence(aspect).reshape(1, -1))
        complete_data = [data, aspect]
        output = model(complete_data)
        sentiment = self.polarity_dict[int(torch.argmax(output, -1))]
        return sentiment


if __name__ == '__main__':
    text = "MS Office 2011 for Mac is wonderful, well worth it."
    aspect = "MS Office 2011 for Mac"
    predictor = Predictor.build_prerequisites()
    sentiment = predictor.execute(text, aspect)
    print(sentiment)
