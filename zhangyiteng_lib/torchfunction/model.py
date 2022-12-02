import torch
import torch.nn as nn
from typing import Tuple
from zytlib import table

class LinearRegression(nn.Module):

    def __init__(self, p, bias=False, l2_reg=0, l1_reg=0, device=None):
        super().__init__()
        self.has_bias = bias
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.linear = nn.Linear(p, 1, bias=bias, device=device)

    def forward(self, x):

        return self.linear(x)

    def fit(self, x, y, n_iter=1000, lr=1e-3, tol=1e-6, verbose=False):
        # x.shape: [N, p]
        # y.shape: [N]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        y = y.view(-1, 1)

        best_loss = 1e100
        step = 0
        count = 0
        best_state = self.linear.state_dict().copy()
        while n_iter < 0 or step < n_iter:
            pred_y = self(x)
            loss = criterion(pred_y, y)
            if self.l2_reg > 0:
                loss += self.l2_reg * self.linear.weight.pow(2).sum()
            if self.l1_reg > 0:
                loss += self.l1_reg * self.linear.weight.abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"epoch {step}, loss {loss.item()}")
            improvement = best_loss - loss.item()
            if improvement < min(1, best_loss) * tol:
                count += 1
                if count > 10:
                    break
            else:
                count = 0
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.linear.state_dict().copy()
            step += 1
        self.linear.load_state_dict(best_state)
        print(f"total epoch {step}")

class rcOCCA(nn.Module):

    def __init__(self, feature_dim, label_dim, k, device=None):
        super().__init__()
        self.hyper = table()
        self.hyper.feature_dim = feature_dim
        self.hyper.label_dim = label_dim
        self.hyper.k = k
        self.pa = nn.Parameter(torch.zeros(feature_dim, k, device=device), requires_grad=False)
        self.pb = nn.Parameter(torch.zeros(label_dim, k, device=device), requires_grad=False)

    @torch.no_grad()
    def fit(self, Xa, Xb):
        # X_a: [n, feature_dim]
        # X_b: [n, label_dim]
        # z_score X_a, X_b
        def _z_score(x: torch.Tensor, dim: int=0) -> torch.Tensor:
            x = x - torch.mean(x, dim=dim, keepdim=True)
            x = x / torch.std(x, dim=dim, keepdim=True)
            return x
        X_a = _z_score(Xa)
        X_b = _z_score(Xb)
        def _svd(x: torch.Tensor, threshold=1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            u, s, vh = torch.linalg.svd(x)
            s = s[s > threshold]
            return u[:, :s.shape[0]], s, vh[:s.shape[0], :].T
        def _rc(xa: torch.Tensor, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            ua, sa, va = _svd(xa)
            ub, sb, vb = _svd(xb)
            u, s, v = _svd(ua.T @ ub)
            if s.shape[0] == 0:
                return None, None
            hat_pa = u[:, :1]
            hat_pb = v[:, :1]
            pa = (va @ torch.diag(1 / sa) @ hat_pa)
            pb = (vb @ torch.diag(1 / sb) @ hat_pb)
            return pa / pa.norm(), pb / pb.norm()
        for i in range(self.hyper.k):
            pa, pb = _rc(X_a, X_b)
            if pa is None:
                print(f"max rank is {i}")
                break
            self.pa.data[:, i] = pa.squeeze()
            self.pb.data[:, i] = pb.squeeze()
            print(f"correlation between {i+1} axis:", torch.corrcoef(torch.cat([Xa @ pa, Xb @ pb], 1).T)[0, 1].item())
            X_a = X_a - (X_a @ pa) @ pa.T
            X_b = X_b - (X_b @ pb) @ pb.T
