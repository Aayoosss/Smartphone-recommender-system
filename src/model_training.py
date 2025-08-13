from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import BaseEstimator
import logging
import pandas as pd
import joblib
import os

log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('model_training.py')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'model_training.log')
fileHandler = logging.FileHandler(file_path)
fileHandler.setLevel('DEBUG')

Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileHandler.setFormatter(Formatter)
consoleHandler.setFormatter(Formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        df = pd.read_csv(path)
        Target = df['Price']
        X_train = df.copy().drop(columns = ['Price'])
        logger.debug("Data Loaded Succesfully")
        return X_train, Target
        
    except Exception as e:
        logger.debug("Unexpected error occurred while loading dataset: %s", e)
        raise
    
def train_model(X_train: pd.DataFrame, y_train: pd.Series, modeltype: str = 'linear') -> BaseEstimator:
    try:
        if modeltype == 'linear':
            reg_model = LinearRegression()
        elif modeltype == 'ridge':
            reg_model = Ridge(alpha = 0.1)
        elif modeltype == 'lasso':
            reg_model = Lasso(alpha=0.1)
        reg_model.fit(X_train, y_train)
        logger.debug("Model Trained Successfully")
        return reg_model
    
    except Exception as e:
        logger.debug("Unexpected error occured while fitting the model: %s", e)
        raise
    
def save_model(model: BaseEstimator, path: str, modeltype: str = 'linear') -> None:
    try:
        os.makedirs(path, exist_ok=True)
        if modeltype == 'linear':
            filename = 'lr_model.pkl'
        elif modeltype == 'ridge':
            filename = 'ridge_model.pkl'
        elif modeltype == 'lasso':
            filename = 'lasso_model.pkl'
            

        os.makedirs(os.path.join(path, modeltype), exist_ok=True)  
        joblib.dump(model, os.path.join(os.path.join(path, modeltype), filename))
        logger.debug("Model Saved Succesfully at %s", os.path.join(path, modeltype))
        
    except Exception as e:
        logger.debug("Unexpected error occurred while saving the model: %s", e)
        raise

def main():
    try:
        train_path = 'data/Processed_data/processed_train.csv'
        X_train, y_train = load_data(train_path)
        modeltype = 'lasso' 
        Model = train_model(X_train, y_train, modeltype)
        model_path = 'models' 
        save_model(Model, model_path, modeltype)
        
    except Exception as e:
        logger.debug("Unexpected error occurred while traing the model: %s", e)
        raise
    
    
if __name__ == '__main__':
    main()
