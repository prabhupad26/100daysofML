import numpy as np
import torch
from tqdm import tqdm
import clip
from dataset import get_data_loader
import matplotlib.pyplot as plt
from PIL import Image
import os


class ClipInference:
    def __init__(self, model_name="ViT-B/32", data="cifar100"):
        self.device = self.get_device()
        self.model, self.loader, \
        self.preprocess, self.classes = self.get_model(model_name, data)
        self.dataset_name = data

    def get_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device

    def image_from_text(self, query):
        # TODO : Train in colab and Save the weights and use it here
        model = self.model
        loader = self.loader
        device = self.device
        prob_max = 0
        prob_max_idx = None
        text_features = self.get_text_encoding(model, query)
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.to(device)
                target = target.to(device)

                print(f"Encoding images batch : {i}")
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                print(f"Creating embedding space with batch : {i}")
                pred = (100. * text_features @ image_features.T).topk(max((1,)), 1, True, True)
                if pred.values > prob_max:
                    prob_max = pred.values
                    prob_max_idx = pred.indices
                    img_feat = images[prob_max_idx]
                break
        return prob_max, img_feat

    def text_from_image(self, input_img='target.jpg'):
        input_img = f"static/{input_img}"
        classes = self.classes
        zeroshot_weights = self.get_pretrained_text()
        test_image = Image.open(input_img)
        test_image = self.preprocess(test_image)
        with torch.no_grad():
            test_image_feat = self.model.encode_image(test_image.reshape(1, 3, 224, 224)).float()

        text_probs = (100.0 * test_image_feat @ zeroshot_weights.float()).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

        plt.figure(figsize=(5, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(input_img))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[0])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        if classes:
            plt.yticks(y, [classes[index] for index in top_labels[0].numpy()])
        else:
            plt.yticks(y, [index for index in top_labels[0].numpy()])
        plt.xlabel("probability")
        file_name = f"prediction_{os.path.basename(input_img)}"
        print(f"Saving image : {file_name}")
        plt.savefig(f"static/{file_name}")
        return file_name

    def get_pretrained_text(self):
        dataset_name = self.dataset_name
        if dataset_name == "imagenetv2":
            path = 'assets/model/zeroshot_weights.pt'
            weights = torch.load(path)
        elif dataset_name == 'cifar100':
            path = 'assets/model/cifar100_weights.pt'
            weights = torch.load(path)
        return weights

    def get_model(self, model_name, data="imagenetv2"):
        print(f"Using pretrained model : {model_name}")
        model, preprocess = clip.load(model_name)
        model.to(self.device).eval()
        loader, classes = get_data_loader(preprocess, data)
        return model, loader, preprocess, classes

    def get_text_encoding(self, model, query):
        print(f"Encoding query : {query}")
        query_tok = clip.tokenize(query)
        with torch.no_grad():
            text_features = model.encode_text(query_tok).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def convert_image(self, image):
        image = image.numpy().squeeze().astype('float64')
        image = np.transpose(image, (1, 2, 0))
        image = image * np.array((0.26862954, 0.26130258, 0.27577711)) \
                + np.array((0.48145466, 0.4578275, 0.40821073))
        image = np.clip(image, 0, 1)
        return image

    def save_image(self, query, probability, img_feat):
        plt.subplot(1, 1, 1)
        plt.title(f"{query}, probability : {probability.cpu().squeeze().numpy().astype('str')}")
        plt.imshow(self.convert_image(img_feat), interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        file_name = f"{query}.png"
        print(f"Saving image : output/{file_name}")
        plt.savefig(f"static/{file_name}")
        return file_name


if __name__ == '__main__':
    query = "a photo of a goldfish"
    model_inference = ClipInference()
    probability, img_feat = model_inference.image_from_text(query)
    model_inference.save_image(query, probability, img_feat)
    print(model_inference.text_from_image())
