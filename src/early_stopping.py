class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        Initialize the EarlyStopping object.

        Parameters:
        - patience: How many epochs to wait before stopping if no improvement is seen.
        - delta: Minimum change in the loss to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def should_stop(self, current_loss):
        """
        Check if training should be stopped based on the current loss.

        Parameters:
        - current_loss: The loss at the current epoch.

        Returns:
        - True if early stopping condition is met, otherwise False.
        """
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


# def train_model(device, n_epochs=10000, patience=50, learning_rate=0.001):
#     model = AngiogenesisPINN(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Training data
#     x = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
#     t = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
#     X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
#     x_train = X.reshape(-1, 1)
#     t_train = T.reshape(-1, 1)

#     # Early stopping
#     early_stopping = EarlyStopping(patience=patience)

#     best_loss = float('inf')
#     for epoch in range(n_epochs):
#         optimizer.zero_grad()

#         pde_l = model.pde_loss(x_train, t_train)
#         bc_l = model.boundary_loss(x_train, t_train)
#         ic_l = model.initial_loss(x_train)

#         loss = pde_l + bc_l + ic_l
#         loss.backward()
#         optimizer.step()

#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             torch.save(model.state_dict(), 'best_model.pth')

#         if early_stopping.should_stop(loss.item()):
#             print(f"Early stopping at epoch {epoch}")
#             break

#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {loss.item():.2e}')

#     return model
