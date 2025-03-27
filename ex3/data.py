from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import price_to_class

def load_data(path: str, target: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if target not in df.columns:
        X, y = df, None
        return X, y
    
    X, y = df.drop(target, axis=1), df[target].map(price_to_class)
    return X, y

def index_encode(df: pd.DataFrame, column: str, mappings: dict = None) -> tuple[pd.DataFrame, dict]:
    if not column in df.columns or df[column].dtype != 'object':
        return df, {}
    
    unique_values = df[column].unique()
    if mappings is not None:
        unique_values = [value for value in unique_values]
        missings = [value for value in unique_values if value not in mappings]
        if len(missings) > 0:
            raise ValueError(f"Missing values in mappings: {missings}")
    else:
        mappings = {value: index for index, value in enumerate(unique_values)}

    encoded_df = df.copy()
    encoded_df[column] = encoded_df[column].map(mappings)

    return encoded_df, mappings

def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if not column in df.columns:
        return df

    cols = pd.get_dummies(df[column], prefix=column)

    encoded_df = pd.concat([df, cols], axis=1)
    encoded_df.drop(column, axis=1, inplace=True)

    return encoded_df


def polynomial_expansion(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
    df_expanded = df.copy()
    subset = df_expanded[columns].values

    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=False,
        include_bias=False
    )
    
    subset_poly = poly.fit_transform(subset)
    new_col_names = poly.get_feature_names_out(columns)
    df_poly = pd.DataFrame(subset_poly, columns=new_col_names, index=df_expanded.index)
    df_expanded.drop(columns=columns, inplace=True)
    df_expanded = pd.concat([df_expanded, df_poly], axis=1)

    return df_expanded


def to_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, class_weights: dict = None) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    sampler = None
    if class_weights is not None:
        y_series = pd.Series(y)
        y_classes = y_series.apply(price_to_class)
        y_weights = y_classes.map(class_weights).values
        sampler = WeightedRandomSampler(
            weights=y_weights,
            num_samples=len(y_weights),
            replacement=True
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    return dataloader