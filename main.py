import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from datasets import load_dataset
import pandas as pd
import optuna

class AdaptiveSpline(nn.Module):
    def __init__(self, num_knots=10, init_range=(-1, 1)):
        super().__init__()
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(init_range[0], init_range[1], num_knots))
        self.coeffs = nn.Parameter(torch.randn(num_knots))
        self.basis_degree = 3

    def forward(self, x):
        x_expanded = x.unsqueeze(-1).expand(-1, self.num_knots)
        basis = self.bspline_basis(x_expanded, self.knots, self.basis_degree)
        return torch.sum(basis * self.coeffs, dim=-1)

    def bspline_basis(self, x, knots, degree):
        m = len(knots) - 1
        n = x.shape[0]
        basis = torch.zeros(n, m - degree, device=x.device)
        
        for j in range(m - degree):
            basis[:, j] = torch.prod(torch.clamp((x - knots[j]) / (knots[j+1:j+degree+1] - knots[j]), 0, 1), dim=1)
        
        return basis

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activations = nn.ModuleList([AdaptiveSpline() for _ in range(in_features * out_features)])

    def forward(self, x):
        outputs = []
        for i in range(self.out_features):
            feature_outputs = torch.stack([self.activations[i * self.in_features + j](x[:, j]) for j in range(self.in_features)])
            outputs.append(torch.sum(feature_outputs, dim=0))
        return torch.stack(outputs, dim=1)

class KAN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(layer_sizes[i], layer_sizes[i+1]) 
                                     for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def visualize_activations(self, top_k=5):
        fig, axes = plt.subplots(len(self.layers), 1, figsize=(12, 4*len(self.layers)))
        if len(self.layers) == 1:
            axes = [axes]
        
        for l, layer in enumerate(self.layers):
            importances = [torch.abs(act.coeffs).sum().item() for act in layer.activations]
            top_indices = np.argsort(importances)[-top_k:]
            
            for idx in top_indices:
                i, j = idx % layer.in_features, idx // layer.in_features
                activation = layer.activations[idx]
                x = torch.linspace(activation.knots.min(), activation.knots.max(), 100)
                y = activation(x).detach().numpy()
                axes[l].plot(x, y, label=f'In {i+1} -> Out {j+1}')
            
            axes[l].set_title(f'Layer {l+1} Top {top_k} Activations')
            axes[l].legend()
            axes[l].grid(True)
        
        plt.tight_layout()
        plt.show()

def kan_loss(model, outputs, targets, l1_weight):
    mse_loss = F.mse_loss(outputs, targets)
    l1_loss = sum(torch.sum(torch.abs(p)) for p in model.parameters())
    return mse_loss + l1_weight * l1_loss

def train_model(model, train_loader, val_loader, epochs, lr, l1_weight, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = kan_loss(model, outputs, y_batch, l1_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += F.mse_loss(outputs, y_val).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    l1_weight = trial.suggest_loguniform('l1_weight', 1e-5, 1e-2)
    hidden_size = trial.suggest_int('hidden_size', 5, 20)
    
    model = KAN([7, hidden_size, hidden_size // 2, 1])
    model, _, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=lr, l1_weight=l1_weight)
    
    return min(val_losses)

# Load and preprocess the data
dataset = load_dataset("mstz/energy_consumption")
df = pd.DataFrame(dataset['train'])

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month

features = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows', 'hour', 'month']
target = 'energy_consumption'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the final model with best hyperparameters
final_model = KAN([7, best_params['hidden_size'], best_params['hidden_size'] // 2, 1])
final_model, train_losses, val_losses = train_model(final_model, train_loader, val_loader, 
                                                    epochs=200, lr=best_params['lr'], 
                                                    l1_weight=best_params['l1_weight'])

# Evaluate the model
final_model.eval()
with torch.no_grad():
    test_predictions = final_model(X_test_tensor)
    test_mse = F.mse_loss(test_predictions, y_test_tensor)
    test_rmse = torch.sqrt(test_mse)

print(f"Test RMSE: {test_rmse.item():.4f}")

# Visualize the learned activation functions
final_model.visualize_activations(top_k=3)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Feature importance analysis
feature_importances = []
for layer in final_model.layers:
    layer_importances = []
    for i in range(layer.in_features):
        importance = sum(torch.abs(act.coeffs).sum() for act in layer.activations[i::layer.in_features])
        layer_importances.append(importance.item())
    feature_importances.append(layer_importances)

# Visualize feature importances
plt.figure(figsize=(12, 6))
for i, importances in enumerate(feature_importances):
    plt.bar(np.arange(len(importances)) + i*0.25, importances, width=0.25, label=f'Layer {i+1}')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances Across Layers')
plt.legend()
plt.xticks(np.arange(len(features)), features, rotation=45)
plt.tight_layout()
plt.show()