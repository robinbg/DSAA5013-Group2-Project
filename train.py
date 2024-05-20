import timm 
from torchvision import datasets, transforms
import torchvision
from transformers import TrainingArguments, Trainer
from utils import Cutout
import argparse
# Define transformations
# Cifar-10 data
MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.2023, 0.1994, 0.2010]
parser = argparse.ArgumentParser(description="Test different models on CIFAR10 dataset.")


transform = [transforms.ToTensor()]
train_transform = transform + [Cutout(1,16)] 
train_transform = transforms.Compose(train_transform)
transform = transforms.Compose(transform)
# Data
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--pretrained", type=bool, required=False)
args = parser.parse_args()
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    max_steps=10000,
    bf16_full_eval=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=1e-3,
    weight_decay=0,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_num_workers=4
)
training_args.set_lr_scheduler('cosine', max_steps=10000)
# Define metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).astype(float).mean().item()}

from model_for_ft import EffNetV2
model = EffNetV2(pretrained_path=args.model_path, pretrained=args.pretrained, num_classes=10)
import torch

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example[0]))
        labels.append(example[1])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"x": pixel_values, "labels": labels}

model = model.to("cuda")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=testset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# Train the model
trainer.train()
