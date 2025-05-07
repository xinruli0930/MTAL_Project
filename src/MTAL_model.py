import torch
import torch.nn as nn
import torch.optim as optim

# Multi-task Outcome Generator
class OutcomeGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks):
        super(OutcomeGenerator, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Individual task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        shared_output = self.shared_layers(x)
        outputs = [head(shared_output) for head in self.task_heads]
        return torch.cat(outputs, dim=1)

# Adversarial True-False Discriminator
class TFDiscriminator(nn.Module):
    def __init__(self, num_tasks, hidden_dim):
        super(TFDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(num_tasks, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, y_pred):
        return self.discriminator(y_pred)

# MTAL model wrapper
class MTALModel:
    def __init__(self, input_dim, hidden_dim, num_tasks, lr=1e-3):
        self.generator = OutcomeGenerator(input_dim, hidden_dim, num_tasks)
        self.discriminator = TFDiscriminator(num_tasks, hidden_dim)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, x, y_true, factual_mask):
        # Generate predicted outcomes
        y_pred = self.generator(x)

        # Discriminator training (True: factual=1, False: counterfactual=0)
        factual_labels = torch.ones((x.size(0), 1))
        cfactual_labels = torch.zeros((x.size(0), 1))

        disc_real_loss = self.bce_loss(self.discriminator(y_true), factual_labels)
        disc_fake_loss = self.bce_loss(self.discriminator(y_pred.detach()), cfactual_labels)
        disc_loss = (disc_real_loss + disc_fake_loss) / 2

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Generator training (tries to fool discriminator)
        gen_adv_loss = self.bce_loss(self.discriminator(y_pred), factual_labels)
        gen_mse_loss = self.mse_loss(y_pred * factual_mask, y_true * factual_mask)
        gen_total_loss = gen_mse_loss + 0.1 * gen_adv_loss  # balance MSE and adversarial loss

        self.gen_optimizer.zero_grad()
        gen_total_loss.backward()
        self.gen_optimizer.step()

        return {
            'disc_loss': disc_loss.item(),
            'gen_loss': gen_total_loss.item()
        }

# Example Usage:
if __name__ == "__main__":
    # Example input dimensions
    input_dim = 25    # number of covariates
    hidden_dim = 64   # hidden layer size
    num_tasks = 3     # number of subgroups (tasks)
    
    # Dummy data (batch size=10)
    x_batch = torch.randn((10, input_dim))
    y_true_batch = torch.randn((10, num_tasks))

    # Factual mask (1 for observed factual outcomes, 0 for counterfactual)
    factual_mask = torch.randint(0, 2, (10, num_tasks)).float()

    # Initialize MTAL model
    mtal = MTALModel(input_dim, hidden_dim, num_tasks)

    # Single training step
    losses = mtal.train_step(x_batch, y_true_batch, factual_mask)

    print(f"Generator Loss: {losses['gen_loss']:.4f}, Discriminator Loss: {losses['disc_loss']:.4f}")
