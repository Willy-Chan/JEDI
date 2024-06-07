"""
Note: This code was written and run in a google colab notebook. It assumes you have the JSON dataset loaded in Google Drive
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
import json
import random

# from google.colab import drive
# drive.mount('/content/drive')
# file_path = '/content/drive/My Drive/masked_examples_LARGE.json'
# ! pip install git+https://github.com/google-research/bleurt.git
# ! wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
# ! unzip bleurt-base-128.zip


# Finetuning Code
file_path = '/content/drive/My Drive/masked_examples_LARGE.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# MODEL_TO_FINETUNE    (bart-base)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = model.to(device)

class DialogueDataset(Dataset):
    def __init__(self, tokenizer, inputs, targets, max_len=512):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        input_encoding = tokenizer(input_text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        target_encoding = tokenizer(target_text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')

        labels = target_encoding['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100     # set padding token id to -100 so that it is ignored in loss computation
        return input_encoding['input_ids'].squeeze(), labels.squeeze()

inputs = [item['input'].replace('<MASK>', tokenizer.mask_token) for item in data]
targets = [item['target'] for item in data]
input_train, input_val, target_train, target_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

train_dataset = DialogueDataset(tokenizer, input_train, target_train)
val_dataset = DialogueDataset(tokenizer, input_val, target_val)

batch_size = 8
num_epochs = 16
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
optimizer = AdamW(model.parameters(), lr=5e-5)

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)

        # AMP: seems to solve memory issues
        with autocast():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        # GRADIENT SCALING: seems to work much better than random subsets!
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {avg_train_loss}")

model.save_pretrained('trained_model')
tokenizer.save_pretrained('trained_tokenizer')
