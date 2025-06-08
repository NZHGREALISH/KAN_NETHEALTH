import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

class DepressionMultiTaskBaseline:
    def __init__(self, alpha=1.0, hidden_dim=64, lr=1e-3, epochs=2000, device=None):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_names = []
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loss_history = []

    def prepare_features_targets(self, df):
        fitbit_features = [col for col in df.columns if col != 'egoid' and 
                          not any(scale in col for scale in ['CESD', 'BDI', 'BAI', 'BigFive', 
                                                            'SelfEsteem', 'STAI', 'PSQI', 'Trust', 
                                                            'SELSA', 'Stress'])]
        scale_features = [
            'BigFive_Extraversion_T1', 'BigFive_Agreeableness_T1', 
            'BigFive_Conscientiousness_T1', 'BigFive_Neuroticism_T1', 'BigFive_Openness_T1',
            'SelfEsteem_T1', 'Trust_T1'
        ]
        fitbit_features = [col for col in fitbit_features if col in df.columns]
        scale_features = [col for col in scale_features if col in df.columns]
        feature_columns = fitbit_features + scale_features
        self.feature_names = feature_columns
        target_columns = {
            'depression_bdi': 'depression_bdi',
            'anxiety_bai': 'anxiety_bai',
            'sleep_psqi': 'sleep_psqi',
            'selfesteem': 'selfesteem'
        }
        available_targets = {}
        for task_name, col_name in target_columns.items():
            if col_name in df.columns:
                available_targets[task_name] = col_name
        self.target_names = list(available_targets.keys())
        X = df[feature_columns].copy()
        y = df[[available_targets[task] for task in self.target_names]].copy()
        y.columns = self.target_names
        return X, y

    class MTLNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(output_dim)])
        def forward(self, x):
            shared = self.shared(x)
            return torch.cat([head(shared) for head in self.heads], dim=1)

    def train_models(self, X, y):
        print("Training MTL neural network...")
        X_filled = X.fillna(X.mean())
        X_scaled = self.scaler.fit_transform(X_filled)
        y_filled = y.fillna(y.mean())
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_filled.values, dtype=torch.float32, device=self.device)
        input_dim = X_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        self.model = self.MTLNet(input_dim, output_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.loss_history = []
        self.target_names = list(y.columns)
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def evaluate_models(self, X, y):
        print("\nModel evaluation results:")
        print("=" * 60)
        X_filled = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X_filled)
        y_filled = y.fillna(y.mean())
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_filled.values, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        results = {}
        for i, task_name in enumerate(self.target_names):
            y_true = y[task_name].values
            y_pred = preds[:, i]
            mask = ~np.isnan(y_true)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            results[task_name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'n_samples': len(y_true)
            }
            print(f"\n{task_name.upper()}:")
            print(f"  Samples: {len(y_true)}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
        return results

    def plot_loss_curve(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    train_file = "/home/grealish/summer/KAN_NET/predict/train.csv"
    test_file = "/home/grealish/summer/KAN_NET/predict/test.csv"
    model = DepressionMultiTaskBaseline(alpha=1.0)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    X_train, y_train = model.prepare_features_targets(train_df)
    X_test, y_test = model.prepare_features_targets(test_df)
    print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    model.train_models(X_train, y_train)
    model.plot_loss_curve()
    print("\n在训练集上的表现:")
    model.evaluate_models(X_train, y_train)
    print("\n在测试集上的表现:")
    model.evaluate_models(X_test, y_test)

def run():
    print("抑郁症状预测多任务学习Baseline模型")
    print("=" * 50)
    main()

if __name__ == "__main__":
    run()