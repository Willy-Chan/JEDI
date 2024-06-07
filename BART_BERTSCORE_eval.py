"""
Gets the metrics for the BERT score: script intended for BART
"""
# !pip install bert-score transformers torch
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from sklearn.model_selection import train_test_split
from bert_score import score

file_path = '/content/drive/My Drive/masked_examples_LARGE.json'
with open(file_path, 'r') as file:
    data = json.load(file)


### LOAD YOUR SAVED MODEL
### DEFAULT NON FINETUNE VARIANT:
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = model.to(device)
model.eval()

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
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100

        return input_encoding['input_ids'].squeeze(), labels

inputs = [item['input'] for item in data]
targets = [item['target'] for item in data]
input_train, input_val, target_train, target_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)
val_dataset = DialogueDataset(tokenizer, input_val, target_val)
val_loader = DataLoader(val_dataset, batch_size=8)

def generate_and_evaluate_bertscore(model, tokenizer, dataloader, device):
    model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model.generate(input_ids, max_length=50)   # gen responses
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # DECODE TOKENS
            decoded_refs = []
            for idx in range(labels.size(0)):
                label_ids = labels[idx]
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                ref_text = tokenizer.decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_refs.append(ref_text)

            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_refs)
    return all_preds, all_refs

pred_texts, ref_texts = generate_and_evaluate_bertscore(model, tokenizer, val_loader, device)
P, R, F1 = score(pred_texts, ref_texts, lang="en", verbose=True)
for p, r, f1 in zip(P, R, F1):
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f1:.4f}")

print(f"Average Precision: {P.mean().item():.4f}")
print(f"Average Recall: {R.mean().item():.4f}")
print(f"Average F1 Score: {F1.mean().item():.4f}")