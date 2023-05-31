# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, in_features):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU()
        )

        # fully connected layer for pretrain task
        self.fc = nn.Sequential(
            nn.Linear(1024, 1)
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.fc(x)
      return x


def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set

    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # Model declaration
    model = Net(in_features)
    model.train()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        # Iterate over the training dataset
        for i in range(0, len(x_tr), batch_size):
            inputs = x_tr[i:i + batch_size]
            targets = y_tr[i:i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model.forward(inputs).flatten()

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Compute validation loss
        with torch.no_grad():
            model.eval()
            for i in range(0, len(x_val), batch_size):
                inputs = x_val[i:i + batch_size]
                targets = y_val[i:i + batch_size]

                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model.forward(inputs).flatten()

                # Compute the loss
                val_loss = criterion(outputs, targets)

                # Update the weights
                optimizer.step()

        # Print the loss for this epoch
        print(f"Epoch {epoch + 1}/{5}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # Convert input to torch tensor
        x = torch.tensor(x, dtype=torch.float)
        # Extract features using the model's encoder
        features = model.encoder(x).detach().numpy()
        return features

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = LinearRegression()
    #model = RandomForestRegressor(n_estimators=100, max_depth=10)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()


    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    pipeline = Pipeline([
        ("pretrain", PretrainedFeatureClass(feature_extractor="pretrain")),
        ("regression", regression_model)
    ])
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    pipeline.fit(x_train, y_train)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
    y_pred = pipeline.predict(x_test_tensor)
    assert y_pred.shape == (x_test.shape[0],)

    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")