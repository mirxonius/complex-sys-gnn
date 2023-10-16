from torchmetrics.regression import RelativeSquaredError, R2Score, MeanAbsoluteError
import torch


class RegressionMetricCalc(torch.nn.Module):
    """
    Simple module used for regression metric calculation
    """

    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.r2_score = R2Score(num_outputs=num_outputs)
        self.rmse = RelativeSquaredError(num_outputs=num_outputs)
        self.mae = MeanAbsoluteError()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, step_type: str
    ) -> dict:
        with torch.no_grad():
            metric_dict = {
                step_type + "_RMSE": self.rmse(y_pred, y_true),
                step_type + "_R2": self.r2_score(y_pred, y_true),
                step_type + "_MAE": self.mae(y_pred, y_true),
            }
            return metric_dict
