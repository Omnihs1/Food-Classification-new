from net.EfficientNet import EfficientNet
from trainer.trainer import Trainer
from data.build import build_dataset
import pytorch_lightning as pl

#call data
train_dataloader, test_dataloader = build_dataset()

# call model
model_name = 'efficientnet_b0'
model = EfficientNet(model_name)


# train model
trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=train_dataloader)