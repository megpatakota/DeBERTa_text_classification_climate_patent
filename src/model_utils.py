# src/model_utils.py

import logging
import torch
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
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
    pr_auc = average_precision_score(labels, logits[:, 1])
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
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def create_datasets(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    test_texts,
    test_labels,
    tokenizer,
    config,
):
    logging.info("Tokenizing training data...")
    train_inputs = tokenize_function(train_texts, tokenizer, config)

    logging.info("Tokenizing validation data...")
    val_inputs = tokenize_function(
        val_texts, tokenizer, config
    )  # Tokenize validation data

    logging.info("Tokenizing testing data...")
    test_inputs = tokenize_function(test_texts, tokenizer, config)

    logging.info("Creating training, validation, and testing datasets...")
    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"],
            "labels": torch.tensor(train_labels),
        }
    )

    val_dataset = Dataset.from_dict(
        {
            "input_ids": val_inputs["input_ids"],
            "attention_mask": val_inputs["attention_mask"],
            "labels": torch.tensor(val_labels),
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "input_ids": test_inputs["input_ids"],
            "attention_mask": test_inputs["attention_mask"],
            "labels": torch.tensor(test_labels),
        }
    )

    return train_dataset, val_dataset, test_dataset  # Return all three datasets


def predict_proba(texts, model, tokenizer, device, config):
    """
    Returns the predicted probabilities for the positive class.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize input texts
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )

    # Move inputs to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ensure the model is on the correct device
    model.to(device)
    model.eval()

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    # Return the probability for the positive class
    return probs[:, 1].cpu().numpy()  # Probability of the positive class
