import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import Resnet18, Resnet12, SimpleCNN, NetExample
from ptl_modules import FaceDataModule, FriendFaceDetector

try:
    os.system("clear")
except:
    pass

pl.seed_everything(42)

# Setting up data module
data_module = FaceDataModule(batch_size=16)
data_module.setup()

trainloader = data_module.train_dataloader()
a = iter(trainloader)
feat, targ = next(a)

print("Feature Shape: ", feat.shape)
print("Target Shape", targ.shape)

# net = Resnet12(data_channels=3, output_size=3)
# net = SimpleCNN(data_channels=3, output_size=3)
net = NetExample(in_channels=3, out_classes=3)

# Defining Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="training_output",
    filename="best-checkpoint",
    save_top_k=2,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

# Log to Tensor Board
logger = TensorBoardLogger("lightning_logs", name="friend-detection")

# Stop trainining if model is not improving
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=50)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=1)

# Model
model = FriendFaceDetector(net, lr=1e-5)

# Defining a Pytorch Lightning Trainer
N_EPOCHS = 50
trainer = pl.Trainer(
    logger=logger,
    enable_progress_bar=True,
    log_every_n_steps=2,
    callbacks=[early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs=N_EPOCHS,
    accelerator="gpu",
)

# train model
trainer.fit(model, datamodule=data_module)

# save model
trainer.save_checkpoint("training_output/friend-detection.ckpt")
