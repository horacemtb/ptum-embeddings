import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import calculate_accuracy


class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path='best_params.pt'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.save_path = save_path  # Path to save the model's best parameters

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_model(model)  # Save model when it's the first time
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(model)  # Save the new best model
            self.counter = 0

    def save_model(self, model):
        """Save the model's state_dict to a file."""
        torch.save(model.state_dict(), self.save_path)
        print(f"Best model saved at {self.save_path}")

    def load_best_model(self, model):
        """Load the best model from the saved file."""
        model.load_state_dict(torch.load(self.save_path))
        print(f"Best model loaded from {self.save_path}")


def train_one_epoch(model, train_loader_mbp, train_loader_nbp, optimizer, device, lambda_nbp=1):

    model.train()
    total_loss = 0
    mbp_total_accuracy = 0
    nbp_total_accuracy = 0
    num_batches = len(train_loader_mbp)

    for mbp_batch, nbp_batch in tqdm(zip(train_loader_mbp, train_loader_nbp), desc="Training"):
        # MBP Task
        mbp_past_sequences, mbp_candidate_sets, mbp_labels, mbp_padding_masks = mbp_batch
        mbp_past_sequences, mbp_candidate_sets, mbp_labels, mbp_padding_masks = (
            mbp_past_sequences.to(device), mbp_candidate_sets.to(device), mbp_labels.to(device), mbp_padding_masks.to(device)
        )
        mbp_predictions = model(mbp_past_sequences, mbp_candidate_sets, mbp_padding_masks)
        loss_mbp = F.cross_entropy(mbp_predictions.view(-1, mbp_predictions.size(-1)), mbp_labels.view(-1))
        mbp_accuracy = calculate_accuracy(mbp_predictions.view(-1, mbp_predictions.size(-1)), mbp_labels.view(-1))

        # NBP Task
        nbp_past_sequences, nbp_candidate_sets, nbp_labels, nbp_padding_masks = nbp_batch
        nbp_past_sequences, nbp_candidate_sets, nbp_labels, nbp_padding_masks = (
            nbp_past_sequences.to(device), nbp_candidate_sets.to(device), nbp_labels.to(device), nbp_padding_masks.to(device)
        )
        nbp_predictions = model(nbp_past_sequences, nbp_candidate_sets, nbp_padding_masks)
        loss_nbp = F.cross_entropy(nbp_predictions.view(-1, nbp_predictions.size(-1)), nbp_labels.view(-1))
        nbp_accuracy = calculate_accuracy(nbp_predictions.view(-1, nbp_predictions.size(-1)), nbp_labels.view(-1))

        # Combine Losses
        loss = loss_mbp + lambda_nbp * loss_nbp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mbp_total_accuracy += mbp_accuracy
        nbp_total_accuracy += nbp_accuracy

    return total_loss / num_batches, mbp_total_accuracy / num_batches, nbp_total_accuracy / num_batches


def evaluate(model, eval_loader_mbp, eval_loader_nbp, device, lambda_nbp=1):

    model.eval()
    total_loss = 0
    mbp_total_accuracy = 0
    nbp_total_accuracy = 0
    num_batches = len(eval_loader_mbp)

    with torch.no_grad():

        for mbp_batch, nbp_batch in tqdm(zip(eval_loader_mbp, eval_loader_nbp), desc="Evaluating"):
            # MBP Task
            mbp_past_sequences, mbp_candidate_sets, mbp_labels, mbp_padding_masks = mbp_batch
            mbp_past_sequences, mbp_candidate_sets, mbp_labels, mbp_padding_masks = (
                mbp_past_sequences.to(device), mbp_candidate_sets.to(device), mbp_labels.to(device), mbp_padding_masks.to(device)
            )
            mbp_predictions = model(mbp_past_sequences, mbp_candidate_sets, mbp_padding_masks)
            loss_mbp = F.cross_entropy(mbp_predictions.view(-1, mbp_predictions.size(-1)), mbp_labels.view(-1))
            mbp_accuracy = calculate_accuracy(mbp_predictions.view(-1, mbp_predictions.size(-1)), mbp_labels.view(-1))

            # NBP Task
            nbp_past_sequences, nbp_candidate_sets, nbp_labels, nbp_padding_masks = nbp_batch
            nbp_past_sequences, nbp_candidate_sets, nbp_labels, nbp_padding_masks = (
                nbp_past_sequences.to(device), nbp_candidate_sets.to(device), nbp_labels.to(device), nbp_padding_masks.to(device)
            )
            nbp_predictions = model(nbp_past_sequences, nbp_candidate_sets, nbp_padding_masks)
            loss_nbp = F.cross_entropy(nbp_predictions.view(-1, nbp_predictions.size(-1)), nbp_labels.view(-1))
            nbp_accuracy = calculate_accuracy(nbp_predictions.view(-1, nbp_predictions.size(-1)), nbp_labels.view(-1))

            # Combine Losses
            loss = loss_mbp + lambda_nbp * loss_nbp

            total_loss += loss.item()
            mbp_total_accuracy += mbp_accuracy
            nbp_total_accuracy += nbp_accuracy

    return total_loss / num_batches, mbp_total_accuracy / num_batches, nbp_total_accuracy / num_batches