import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, classification_report
from tqdm import tqdm
from model import TextRCNN_Bert
from load_data import traindataloader, valdataloader

#train
def cal_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred_pos[:,:1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), 1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), 1)
    neg_loss = torch.logsumexp(y_pred_neg, 1)
    pos_loss = torch.logsumexp(y_pred_pos, 1)
    loss = torch.mean(neg_loss+pos_loss)
    return loss


SAVED_DIR = './saved_model'
EPOCHS = 5
BERT_PATH = '../bert-base-chinese'
WARMUP_PROPORTION = 0.1
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
model = TextRCNN_Bert.from_pretrained(BERT_PATH)
model.to(device)
total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)
loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        tokens_ids, mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        model.zero_grad()
        out = model(tokens_ids, mask)
        loss = cal_loss(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))
model.save_pretrained(SAVED_DIR)
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)

#eval
model = TextRCNN_Bert.from_pretrained(SAVED_DIR)
model.to(device)
model.eval()
pred_y = np.empty((0, 65))
true_y = np.empty((0, 65))
with torch.no_grad():      
    for batch in valdataloader:    
        tokens_ids, mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        logits = model(tokens_ids, mask)
        pred = torch.where(logits>0, 1, 0)
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        pred_y = np.append(pred_y, pred, axis=0)
        true_y = np.append(true_y, label, axis=0)
print(classification_report(true_y, pred_y, digits=4))
h_loss = hamming_loss(true_y, pred_y)
print(h_loss)