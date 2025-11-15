from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class AccuracyAnalyzer:
    """
    Analyzes accuracy of predicted values against measured ground truth data.
    
    Calculates metrics like MAE, RMSE, and R² for each target attribute.
    """

    def calculate_accuracy(
        self,
        df_predicted: pd.DataFrame,
        df_measured: pd.DataFrame,
        target_attributes: List[str],
        measured_columns: Optional[List[str]] = None,
        join_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate accuracy metrics for predicted vs measured values.
        
        Args:
            df_predicted: DataFrame with predicted values (contains target_attributes columns)
            df_measured: DataFrame with measured/ground truth values
            target_attributes: List of column names with predicted values
            measured_columns: List of column names with measured values (defaults to target_attributes)
            join_key: Optional column name to join dataframes (if indices don't match)
            
        Returns:
            Dictionary mapping each target attribute to its accuracy metrics:
            {
                "attribute_name": {
                    "mae": float,
                    "rmse": float,
                    "r2": float,
                    "n_samples": int
                }
            }
        """
        if measured_columns is None:
            measured_columns = target_attributes

        if len(target_attributes) != len(measured_columns):
            raise ValueError(
                f"Number of target_attributes ({len(target_attributes)}) must match "
                f"number of measured_columns ({len(measured_columns)})"
            )

        # Join dataframes if needed
        if join_key:
            df_merged = pd.merge(
                df_predicted,
                df_measured,
                on=join_key,
                suffixes=("_pred", "_meas"),
            )
        else:
            # Try to align by index
            df_merged = df_predicted.copy()
            for col in measured_columns:
                if col in df_measured.columns:
                    df_merged[f"{col}_meas"] = df_measured[col].values

        results = {}

        for pred_col, meas_col in zip(target_attributes, measured_columns):
            # Get column names
            if join_key:
                pred_col_name = f"{pred_col}_pred"
                meas_col_name = f"{meas_col}_meas"
            else:
                pred_col_name = pred_col
                meas_col_name = f"{meas_col}_meas" if meas_col in df_measured.columns else meas_col

            # Check if columns exist
            if pred_col_name not in df_merged.columns:
                print(f"Warning: Predicted column '{pred_col_name}' not found. Skipping.")
                continue

            if meas_col_name not in df_merged.columns:
                print(f"Warning: Measured column '{meas_col_name}' not found. Skipping.")
                continue

            # Extract values, removing NaN
            predicted = df_merged[pred_col_name].dropna()
            measured = df_merged[meas_col_name].dropna()

            # Align indices
            common_idx = predicted.index.intersection(measured.index)
            if len(common_idx) == 0:
                print(f"Warning: No matching indices for '{pred_col}'. Skipping.")
                continue

            predicted_aligned = predicted.loc[common_idx]
            measured_aligned = measured.loc[common_idx]

            # Remove any remaining NaN values
            mask = ~(pd.isna(predicted_aligned) | pd.isna(measured_aligned))
            predicted_clean = predicted_aligned[mask]
            measured_clean = measured_aligned[mask]

            if len(predicted_clean) == 0:
                print(f"Warning: No valid data points for '{pred_col}'. Skipping.")
                continue

            # Calculate metrics
            mae = mean_absolute_error(measured_clean, predicted_clean)
            rmse = np.sqrt(mean_squared_error(measured_clean, predicted_clean))
            r2 = r2_score(measured_clean, predicted_clean)

            results[pred_col] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "n_samples": int(len(predicted_clean)),
            }

        return results

    def print_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print accuracy results in a formatted way.
        
        Args:
            results: Results dictionary from calculate_accuracy
        """
        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 60)
        print("Accuracy Analysis Results")
        print("=" * 60)

        for attr, metrics in results.items():
            print(f"\n{attr}:")
            print(f"  Samples: {metrics['n_samples']}")
            print(f"  MAE (Mean Absolute Error): {metrics['mae']:.4f}")
            print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
            print(f"  R² (Coefficient of Determination): {metrics['r2']:.4f}")

        print("\n" + "=" * 60)

    def to_dataframe(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Convert results dictionary to a pandas DataFrame.
        
        Args:
            results: Results dictionary from calculate_accuracy
            
        Returns:
            DataFrame with attributes as index and metrics as columns
        """
        if not results:
            return pd.DataFrame()

        data = {
            attr: {
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R²": metrics["r2"],
                "N_Samples": metrics["n_samples"],
            }
            for attr, metrics in results.items()
        }

        return pd.DataFrame(data).T


def main():
    """
    Example usage of AccuracyAnalyzer.
    """
    # Example: Load predicted and measured data
    # df_predicted = pd.read_csv("./data/analysed/predictions.csv")
    # df_measured = pd.read_csv("./data/raw/AdExamples.csv")

    # analyzer = AccuracyAnalyzer()
    # results = analyzer.calculate_accuracy(
    #     df_predicted=df_predicted,
    #     df_measured=df_measured,
    #     target_attributes=["CTR_llm", "CVR_llm"],
    #     measured_columns=["CTR", "CVR"],
    # )

    # analyzer.print_results(results)
    # df_results = analyzer.to_dataframe(results)
    # print("\nResults as DataFrame:")
    # print(df_results)

    print("AccuracyAnalyzer is ready to use. See docstring for usage examples.")


if __name__ == "__main__":
    main()

