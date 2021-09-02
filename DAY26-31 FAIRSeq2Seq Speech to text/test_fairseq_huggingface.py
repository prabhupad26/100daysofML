from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset


model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-mustc-en-fr-st")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-mustc-en-fr-st")

