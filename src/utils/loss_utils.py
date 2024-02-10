import torch


class MSE_MAE_Loss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        """
        This loss combines the mean squared error
        and mean absolute error in order to penalze the
        model at larger and smaller scales.
        Args:
            alpha: float : Scaing factor which multiplies the
                MAE component of the loss.
        """
        super().__init__()
        self.alpha = alpha
        self.MSE = torch.nn.MSELoss()
        self.MAE = torch.nn.L1Loss()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.MSE(y_true, y_pred) + self.alpha * self.MAE(y_true, y_pred)

class MSEScalarLoss():
    def __init__(self,alpha:float) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss(y_true,y_pred)
        norm_true = y_true.norm(dim=-1,keepdim=True)
        norm_pred = y_pred.norm(dim=-1,keepdim=True)
        scalar = 1 - torch.bmm(y_pred.view(-1,1,3),y_true.view(-1,3,1)) / (norm_pred*norm_true)**2
        scalar = scalar.mean()
        return mse_loss + self.alpha*scalar

class HuberScalarLoss(torch.nn.Module):
    def __init__(self,alpha:float) -> None:
        super().__init__()
        self.alpha = alpha
        self.huber = torch.nn.HuberLoss()
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(y_true,y_pred)
        norm_true = y_true.norm(dim=-1,keepdim=True)
        norm_pred = y_pred.norm(dim=-1,keepdim=True)
        scalar = 1 - torch.bmm(y_pred.view(-1,1,3),y_true.view(-1,3,1)) / (norm_pred*norm_true)**2
        scalar = scalar.mean()
        return huber_loss + self.alpha*scalar