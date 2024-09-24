# src/train.py

import logging
import torch
from transformers import Trainer, TrainingArguments
from src.model_utils import compute_metrics
import torch.nn.functional as F
from tqdm import tqdm


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(focal_loss)


def get_training_arguments(config):
    return TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        use_cpu=not torch.cuda.is_available(),  # Automatically set to True if no GPU is available
        logging_steps=config["training"]["logging_steps"],
        logging_first_step=config["training"]["logging_first_step"],
    )


def train_model(model, train_dataset, test_dataset, training_args):
    logging.info("Initializing the Trainer...")

    # Define custom Trainer class to handle focal loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Use your Focal Loss implementation here
            focal_loss = FocalLoss(alpha=0.25)  # Adjust alpha if needed
            loss = focal_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Use tqdm to display a progress bar during training
    logging.info("Starting training with progress bar...")
    progress_bar = tqdm(
        total=training_args.num_train_epochs, desc="Training Progress", unit="epoch"
    )

    try:
        trainer.train()
        for epoch in range(training_args.num_train_epochs):
            progress_bar.update(1)
        progress_bar.close()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e

    return trainer
