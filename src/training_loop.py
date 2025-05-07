import torch
from torch.utils.data import DataLoader, TensorDataset

# Example training function
def train_adversarial_network(generator, discriminator, 
                              train_loader, val_loader, 
                              num_epochs=20, lr=1e-3, device='cpu'):
    
    # Optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Loss functions
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()

    # Move models to device
    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0

        for x_batch, y_true_batch, factual_mask in train_loader:
            x_batch = x_batch.to(device)
            y_true_batch = y_true_batch.to(device)
            factual_mask = factual_mask.to(device)

            # Generate synthetic outcomes
            y_pred = generator(x_batch)

            # --- Discriminator Training ---
            disc_optimizer.zero_grad()

            # Real outcomes (label=1), Fake outcomes (label=0)
            real_labels = torch.ones((x_batch.size(0), 1), device=device)
            fake_labels = torch.zeros((x_batch.size(0), 1), device=device)

            # Compute discriminator loss
            disc_real_loss = bce_loss(discriminator(y_true_batch), real_labels)
            disc_fake_loss = bce_loss(discriminator(y_pred.detach()), fake_labels)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            disc_loss.backward()
            disc_optimizer.step()

            # --- Generator Training ---
            gen_optimizer.zero_grad()

            # Adversarial loss (Generator tries to fool discriminator)
            gen_adv_loss = bce_loss(discriminator(y_pred), real_labels)

            # MSE loss (predicting factual outcomes)
            gen_mse_loss = mse_loss(y_pred * factual_mask, y_true_batch * factual_mask)

            # Total generator loss
            gen_loss = gen_mse_loss + 0.1 * gen_adv_loss  # weight adversarial loss

            gen_loss.backward()
            gen_optimizer.step()

            # Accumulate losses
            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()

        # Average losses per epoch
        epoch_gen_loss /= len(train_loader)
        epoch_disc_loss /= len(train_loader)

        # Validation step (optional but recommended)
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, val_mask in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_mask = val_mask.to(device)
                y_val_pred = generator(x_val)
                loss = mse_loss(y_val_pred * val_mask, y_val * val_mask)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Generator Loss: {epoch_gen_loss:.4f}, "
              f"Discriminator Loss: {epoch_disc_loss:.4f}, Validation Loss: {val_loss:.4f}")

    print("Training complete.")

# Example usage:
if __name__ == "__main__":
    # Example dimensions (adjust according to your dataset)
    input_dim = 25
    hidden_dim = 64
    num_tasks = 3

    # Dummy data for demonstration (replace with your real data)
    X_train = torch.randn(500, input_dim)
    Y_train = torch.randn(500, num_tasks)
    factual_mask_train = torch.randint(0, 2, (500, num_tasks)).float()

    X_val = torch.randn(100, input_dim)
    Y_val = torch.randn(100, num_tasks)
    factual_mask_val = torch.randint(0, 2, (100, num_tasks)).float()

    # Data loaders
    train_dataset = TensorDataset(X_train, Y_train, factual_mask_train)
    val_dataset = TensorDataset(X_val, Y_val, factual_mask_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize models (use classes from previous implementation)
    generator = OutcomeGenerator(input_dim, hidden_dim, num_tasks)
    discriminator = TFDiscriminator(num_tasks, hidden_dim)

    # Train the models
    train_adversarial_network(generator, discriminator, train_loader, val_loader, num_epochs=20, device='cpu')
