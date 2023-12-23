import torch
from monai.networks.nets import UNet
import torch.nn.functional as F
from monai.losses import DiceLoss
import numpy as np
from src.visualization.visualize import view_slice


class UNet(UNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = F.softmax(x)
        return x


from src.utils import numpy_from_tensor

ROOT_DIR = "models/"


def train_network(
    training_loader,
    val_loader,
    network,
    loss_fun,
    optimizer,
    device,
    EPOCHS=200,
    pre_load_training=False,
    checkpoint_name="",
):
    # network.cuda(device)

    optimizer = optimizer
    loss_fun = loss_fun

    epoch_checkpoint = 0

    losses = {}
    val_losses = {}

    # Test Learning rate dictionary for visualization
    scheduler_learning_rate_dict = {}

    if pre_load_training:
        checkpoint = torch.load(
            ROOT_DIR + f"{checkpoint_name}.pt", map_location=torch.device("cpu")
        )
        epoch_checkpoint = checkpoint["epoch"] + 1
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]
        losses = checkpoint["losses"]
        val_losses = checkpoint["val_losses"]
        # learning_rate = checkpoint['learning_rate'] if 'learning_rate' in checkpoint.keys() else optimizer.param_groups[0]["lr"]

    # if epoch_checkpoint > 300:
    #     print(f'Learning rate changed from {optimizer.param_groups[0]["lr"]} to:')
    #     optimizer.param_groups[0]["lr"] = 0.0001
    #     print(f'{optimizer.param_groups[0]["lr"]}')

    # Train the network
    for epoch in range(epoch_checkpoint, EPOCHS):
        network.train(True)

        print(f"losses: {losses}")
        print(f"val losses {val_losses}")

        train_step = 1
        batch_loss = []

        for batch_data in training_loader:
            print(f"Epoch {epoch}\tTraining Step: {train_step}/{len(training_loader)}")

            # torch.cuda.empty_cache() # Clear any unused variables
            inputs = batch_data["image"].to(device)
            labels = batch_data[
                "label"
            ]  # Only pass to CUDA when required - preserve memory

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Feed input data into the network to train
            outputs = network(inputs)

            # Input no longer in use for current iteration - clear from CUDA memory
            inputs = inputs.cpu()
            # torch.cuda.empty_cache()

            # labels to CUDA
            labels = batch_data["label"].to(device)
            # torch.cuda.empty_cache()

            # Calculate DICE CE loss, permute tensors to correct dimensions
            loss = loss_fun(outputs, labels)

            # List of losses for current batch
            batch_loss.append(loss.detach().cpu().numpy())

            # Clear CUDA memory
            labels = labels.cpu()
            # torch.cuda.empty_cache()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            train_step += 1

        # Get average loss for current batch
        losses[epoch] = np.mean(batch_loss)
        print(f"train losses {batch_loss} \nmean loss {losses[epoch]}")

        if epoch % 2 == 0:
            # Set network to eval mode
            network.train(False)
            # Disiable gradient calculation and optimise memory
            with torch.no_grad():
                # Initialise validation loss
                dice_test_loss = 0
                val_iter_count = 0
                # dice_test_loss = []
                for i, batch_data in enumerate(val_loader):
                    # Get inputs and labels from validation set
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"]

                    outputs = network(inputs)

                    # Memory optimization
                    inputs = inputs.cpu()
                    # torch.cuda.empty_cache()
                    labels = batch_data["label"].to(device)

                    # Accumulate DICE CE loss validation error
                    test_loss = loss_fun(outputs, labels)
                    dice_test_loss += test_loss
                    # dice_test_loss += loss_fun(outputs, labels)
                    val_iter_count += 1
                    print(f"Val loss iter {i}: {test_loss}")

                # Get average validation DICE CE loss
                val_losses[epoch] = dice_test_loss / val_iter_count

                # Print errors
                print(
                    "==== Epoch: "
                    + str(epoch)
                    + " | DICE loss: "
                    + str(numpy_from_tensor((dice_test_loss) / val_iter_count))
                    + " | Total Loss: "
                    + str(numpy_from_tensor((dice_test_loss) / val_iter_count))
                    + " ====="
                )  # This is redundant code but will keep here incase we add more losses

                # View slice at halfway point
                half = outputs.shape[2] // 2

                # Show predictions for current iteration
                print(f"shape is {inputs.shape}")
                view_slice(
                    numpy_from_tensor(inputs[0, 0, :, :]),
                    f"Input Channel 0 Image Epoch {epoch}",
                    gray=True,
                )
                view_slice(
                    numpy_from_tensor(inputs[0, 1, :, :]),
                    f"Input Channel 1 Image Epoch {epoch}",
                    gray=True,
                )
                view_slice(
                    numpy_from_tensor(outputs[0, 1, :, :]),
                    f"WMH Output Image Epoch {epoch}",
                    gray=True,
                )
                view_slice(
                    numpy_from_tensor(labels[0, 1, :, :]),
                    f"WMH Labels  Epoch {epoch}",
                    gray=True,
                )

        # Save training checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "losses": losses,
                "val_losses": val_losses,
                # 'learning_rate': optimizer.param_groups[0]['lr']
                # 'scheduler_learning_rate_dict':scheduler_learning_rate_dict
            },
            ROOT_DIR + f"{checkpoint_name}.pt",
        )

        # Confirm current epoch trained params are saved
        print(f"Saved for epoch {epoch}")

    return network


device = torch.device("cpu")
model = UNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.2,
    # kernel_size=3,
).to(device)

loss_function = DiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    1e-3,
    weight_decay=1e-5,
    amsgrad=True,
)

# from torchvision import models
# from torchsummary import summary

# summary(model, (2, 256, 256))

# net = train_network(training_loader=train_loader, val_loader=validation_loader, network=model,
#                     loss_fun=loss_function, optimizer=optimizer, device=device, EPOCHS=2000, pre_load_training=True, checkpoint_name='unet_brain_wmi_UbSi_new_norm')
