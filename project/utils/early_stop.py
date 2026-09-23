import numpy as np
import torch


class EarlyStopping:

    def __init__(
        self, patience=15, min_delta=1e-4, checkpoint_path="best_model.pt"
    ):
        """
        Args:
            patience (int): Quante epoche aspettare senza miglioramenti prima di fermarsi.
            min_delta (float): Miglioramento minimo rispetto alla migliore loss precedente per essere considerato valido.
            checkpoint_path (str): Percorso dove salvare i pesi del miglior modello.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(
                f"⚠️ EarlyStopping counter: {self.counter}/{self.patience} (Migliore: {self.best_loss:.6f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(
                f"✅ Val loss migliorata ({self.best_loss:.6f} --> {val_loss:.6f}). Salvataggio modello..."
            )
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Salva i pesi del modello quando la val_loss migliora."""
        torch.save(model.state_dict(), self.checkpoint_path)