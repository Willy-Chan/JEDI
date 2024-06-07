"""
Eval for DialogRPT, BERT variant
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer_rpt = AutoTokenizer.from_pretrained("microsoft/DialogRPT-human-vs-machine")
# model_rpt = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-human-vs-machine")
tokenizer_rpt = AutoTokenizer.from_pretrained("microsoft/DialogRPT-human-vs-rand")
model_rpt = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-human-vs-rand")
model_rpt.to('cuda') 

def generate_responses(model, tokenizer, dataloader, device):
    model.eval()
    responses = []
    contexts = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            context = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            outputs = model.generate(input_ids, max_length=50)
            decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(decoded_responses)
            contexts.extend(context)

    return contexts, responses

def drpt_eval(model_rpt, tokenizer_rpt, contexts, responses, device):
    model_rpt.eval()
    scores = []
    with torch.no_grad():
        for context, response in zip(contexts, responses):
            inputs = tokenizer_rpt.encode_plus(context, response, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure tensor is on the correct device
            outputs = model_rpt(**inputs)
            score = torch.sigmoid(outputs.logits).squeeze().item()  # Use sigmoid if the logits are not already probabilities
            scores.append(score)
    return scores

contexts, responses = generate_responses(model, tokenizer, val_loader, device)
scores = drpt_eval(model_rpt, tokenizer_rpt, contexts, responses, device)

print("Average DialogRPT Score:", sum(scores) / len(scores))