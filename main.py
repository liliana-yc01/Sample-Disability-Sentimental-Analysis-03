from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    toxic_score = scores[0][1].item()
    return toxic_score

texts = ["Paralympian Ali Truwit lost her leg but still inspires us all."]
for t in texts:
    print(t, predict_toxicity(t))