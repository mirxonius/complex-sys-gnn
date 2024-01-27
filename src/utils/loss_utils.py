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
