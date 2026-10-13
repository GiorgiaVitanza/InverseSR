import copy
import torch


class LatentEarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, save_to_disk=False, checkpoint_path="best_latents.pth"):
        """
        Early stopping specifico per la Latent Optimization / Inversion.
        
        Args:
            patience (int): Numero di step senza miglioramento prima di fermarsi.
            min_delta (float): Soglia minima di miglioramento.
            save_to_disk (bool): Se True salva su disco, altrimenti mantiene i migliori latenti in RAM.
            checkpoint_path (str): Percorso file dove salvare se save_to_disk=True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_to_disk = save_to_disk
        self.checkpoint_path = checkpoint_path
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
        # Dizionario per mantenere i latenti migliori in RAM
        self.best_state = {}

    def __call__(self, loss, latent_variable, cond=None):
        loss_val = float(loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss)
        
        if self.best_loss is None:
            self.best_loss = loss_val
            self._save_best(latent_variable, cond)
        elif loss_val > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience} (Migliore: {self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"✅ Loss migliorata ({self.best_loss:.6f} --> {loss_val:.6f}). Salvataggio stato...")
            self.best_loss = loss_val
            self._save_best(latent_variable, cond)
            self.counter = 0

    def _save_best(self, latent_variable, cond):
        """Salva latent_variable e cond (in RAM o su disco)."""
        state = {
            "latent_variable": latent_variable.detach().clone(),
            "cond": cond.detach().clone() if cond is not None else None,
        }
        
        if self.save_to_disk:
            torch.save(state, self.checkpoint_path)
        else:
            self.best_state = state

    def restore_best(self, latent_variable, cond=None):
        """Ripristina i migliori valori trovati direttamente sui tensori originali."""
        if self.save_to_disk:
            state = torch.load(self.checkpoint_path, map_location=latent_variable.device)
        else:
            state = self.best_state

        if "latent_variable" in state and state["latent_variable"] is not None:
            latent_variable.data.copy_(state["latent_variable"])
        if cond is not None and "cond" in state and state["cond"] is not None:
            cond.data.copy_(state["cond"])
            
        print(f"🔄 Ripristinati latent_variable e cond con la migliore loss ottenuta ({self.best_loss:.6f}).")