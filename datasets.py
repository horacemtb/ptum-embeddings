import torch
from torch.utils.data import Dataset
import random


# PTUMDataset for MBP task
class PTUMMBPDataset(Dataset):
    def __init__(self, user_ids, user_behaviors, num_animes, P):
        self.user_ids = user_ids
        self.user_behaviors = user_behaviors
        self.num_animes = num_animes
        self.P = P  # Number of negative samples

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        behaviors = self.user_behaviors[user_id]

        # Randomly mask one behavior
        masked_idx = random.randint(0, len(behaviors) - 1)
        masked_anime = behaviors[masked_idx]

        # Create negative samples
        candidate_animes = []
        observed_behaviors = set(behaviors)
        while len(candidate_animes) < self.P:
            negative_sample = random.randint(0, self.num_animes - 1)
            if negative_sample not in observed_behaviors:
                candidate_animes.append(negative_sample)

        # Randomly insert the positive sample (masked anime)
        positive_index = random.randint(0, self.P)
        candidate_animes.insert(positive_index, masked_anime)

        # Exclude the masked anime from the behavior sequence
        behaviors_excluding_masked = behaviors[:masked_idx] + behaviors[masked_idx + 1:]

        # Return the user behavior sequence, candidate set, and the positive index (label)
        return behaviors_excluding_masked, torch.tensor(candidate_animes, dtype=torch.long), positive_index


def mbp_collate_fn(batch, max_seq_len=100):
    """
    Collate function for MBP task to pad/truncate sequences and generate masks.
    """
    behavior_sequences, candidate_sets, labels = zip(*batch)

    # Pad or truncate behavior sequences to max_seq_len
    padded_sequences = [
        torch.tensor(seq[:max_seq_len] + [0] * (max_seq_len - len(seq)), dtype=torch.long)
        for seq in behavior_sequences
    ]

    # Generate padding masks
    padding_masks = [
        torch.tensor([1] * min(len(seq), max_seq_len) + [0] * (max_seq_len - min(len(seq), max_seq_len)),
                     dtype=torch.bool)
        for seq in behavior_sequences
    ]

    # Stack tensors
    return (
        torch.stack(padded_sequences),          # Behavior sequences [batch_size, max_seq_len]
        torch.stack(candidate_sets),            # Candidate sets [batch_size, P+1]
        torch.tensor(labels, dtype=torch.long), # Labels (positive index)
        torch.stack(padding_masks)              # Padding masks [batch_size, max_seq_len]
    )


# PTUMDataset for NBP task
class PTUMNBPDataset(Dataset):
    def __init__(self, user_ids, user_behaviors, num_animes, P, K):
        self.user_ids = user_ids
        self.user_behaviors = user_behaviors
        self.num_animes = num_animes
        self.P = P  # Number of negative samples per position
        self.K = K  # Number of future behaviors to predict

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        behaviors = self.user_behaviors[user_id]

        # Ensure the user has enough behaviors for N + K
        if len(behaviors) < self.K + 1:
            raise ValueError(f"User {user_id} has fewer than {self.K + 1} behaviors.")

        # Split into past behaviors and next K behaviors
        past_behaviors = behaviors[:-self.K]
        future_behaviors = behaviors[-self.K:]

        # Create negative samples for each position in K
        candidate_sets = []
        labels = []
        observed_behaviors = set(behaviors)
        for future_behavior in future_behaviors:
            candidates = []
            while len(candidates) < self.P:
                negative_sample = random.randint(0, self.num_animes - 1)
                if negative_sample not in observed_behaviors:
                    candidates.append(negative_sample)

            # Randomly insert the positive behavior
            positive_index = random.randint(0, self.P)
            candidates.insert(positive_index, future_behavior)

            candidate_sets.append(torch.tensor(candidates, dtype=torch.long))
            labels.append(positive_index)

        return (
            torch.tensor(past_behaviors, dtype=torch.long),  # Past behaviors
            torch.stack(candidate_sets),  # Candidate sets for K positions
            torch.tensor(labels, dtype=torch.long)  # Labels (positive indices)
        )


def nbp_collate_fn(batch, max_seq_len=100):
    """
    Collate function for NBP task.
    """
    past_sequences, candidate_sets, labels = zip(*batch)

    # Pad or truncate past behavior sequences

    padded_sequences = [
        torch.cat([seq[:max_seq_len], torch.zeros(max_seq_len - len(seq), dtype=torch.long)])
        if len(seq) < max_seq_len
        else seq[:max_seq_len]
        for seq in past_sequences
        ]

    # Generate padding masks
    padding_masks = [
        torch.tensor([1] * min(len(seq), max_seq_len) + [0] * (max_seq_len - min(len(seq), max_seq_len)),
                     dtype=torch.bool)
        for seq in past_sequences
    ]

    # Stack tensors
    return (
        torch.stack(padded_sequences),              # Past sequences [batch_size, max_seq_len]
        torch.stack(candidate_sets),                # Candidate sets [batch_size, K, P+1]
        torch.stack(labels),                        # Labels [batch_size, K]
        torch.stack(padding_masks)                  # Padding masks [batch_size, max_seq_len]
    )