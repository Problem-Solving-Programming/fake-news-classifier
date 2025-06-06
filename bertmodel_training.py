import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ["WANDB_DISABLED"] = "true"

# 1. 저장된 텐서 불러오기
input_ids = torch.load('input_ids.pt')
attention_mask = torch.load('attention_mask.pt')

# 2. 라벨 불러오기
labels = pd.read_csv("output.csv")["label"].tolist()
labels = torch.tensor(labels)

# 3. 학습/검증 데이터 분리
train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

train_dataset = {
    "input_ids": input_ids[train_idx],
    "attention_mask": attention_mask[train_idx],
    "labels": labels[train_idx]
}
val_dataset = {
    "input_ids": input_ids[val_idx],
    "attention_mask": attention_mask[val_idx],
    "labels": labels[val_idx]
}

# 4. Dataset 클래스 정의
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings["input_ids"][idx],
            'attention_mask': self.encodings["attention_mask"][idx],
            'labels': self.encodings["labels"][idx]
        }
    def __len__(self):
        return len(self.encodings["labels"])

train_dataset = BERTDataset(train_dataset)
val_dataset = BERTDataset(val_dataset)

# 5. 모델 준비
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  
    gradient_accumulation_steps=1  
)

# 7. Trainer로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 8. 모델 저장
model.save_pretrained("./bert-fake-news-model")
