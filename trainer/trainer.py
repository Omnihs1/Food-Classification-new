import pytorch_lightning as pl
import torch.nn as nn

class Trainer(pl.LightningModule):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.model(x)
        loss = nn.CrossEntropyLoss()
        output = loss(input, target)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer