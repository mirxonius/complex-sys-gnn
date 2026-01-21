"""
Regression metrics for model evaluation.

Computes R², RMSE, and MAE for predictions.
"""

from torchmetrics.regression import RelativeSquaredError, R2Score, MeanAbsoluteError
import torch


class RegressionMetricCalc(torch.nn.Module):
    """
    Regression metric calculator module.

    Computes standard regression metrics: R² score, RMSE, and MAE.
    """

    def __init__(self, num_outputs: int = 1):
        """
        Initialize metric calculators.

        Args:
            num_outputs: Number of output dimensions (1 for scalars, 3 for 3D vectors)
        """
        super().__init__()
        self.r2_score = R2Score(num_outputs=num_outputs)
        self.rmse = RelativeSquaredError(num_outputs=num_outputs)
        self.mae = MeanAbsoluteError()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, step_type: str
    ) -> dict:
        """
        Calculate metrics for predictions.

        Args:
            y_pred: Model predictions
            y_true: Ground truth values
            step_type: Prefix for metric names (e.g., 'train', 'valid', 'test')

        Returns:
            Dictionary of computed metrics
        """
        with torch.no_grad():
            metric_dict = {
                step_type + "_RMSE": self.rmse(y_pred, y_true),
                step_type + "_R2": self.r2_score(y_pred, y_true),
                step_type + "_MAE": self.mae(y_pred, y_true),
            }
            return metric_dict
