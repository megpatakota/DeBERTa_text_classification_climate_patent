# src/model_utils.py

import logging
import torch
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from datasets import Dataset
import numpy as np


def tokenize_function(texts, tokenizer, config):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    # Compute ROC-AUC, but only if both classes are present
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(
            labels, logits[:, 1]
        )  # Only if both classes are present
    else:
        roc_auc = None  # Cannot compute ROC-AUC with only one class

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,  # This will be None if only one class is present
    }


def create_datasets(
    train_texts, train_labels, test_texts, test_labels, tokenizer, config
):
    logging.info("Tokenizing training data...")
    train_inputs = tokenize_function(train_texts, tokenizer, config)
    logging.info("Tokenizing testing data...")
    test_inputs = tokenize_function(test_texts, tokenizer, config)
    logging.info("Creating training and testing datasets...")
    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"],
            "labels": torch.tensor(train_labels),
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "input_ids": test_inputs["input_ids"],
            "attention_mask": test_inputs["attention_mask"],
            "labels": torch.tensor(test_labels),
        }
    )
    return train_dataset, test_dataset


def predict_proba(texts, model, tokenizer, device, config):
    """
    Returns the predicted probabilities for the positive class.
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1].cpu().numpy()  # Probability of the positive class
