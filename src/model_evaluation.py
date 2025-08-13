from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import BaseEstimator
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
import joblib
import logging
import os
import json
import pandas as pd


log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)
logger = logging.getLogger('model_evaluation.py')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

file_path = os.path.join(log_dir, "model_evaluation.log")
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
        X_test = df.copy().drop(columns=['Price'])
        y_test = df['Price']
        logger.debug("Data Loaded Successfully")
        return X_test, y_test
    except Exception as e:
        logger.debug("Unexpected error occurred while loading the dataset: %s", e)
        raise
    
def evaluate_model(X_test: pd.DataFrame, y_test: pd.Series, model: BaseEstimator) -> dict:
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2_scr = r2_score(y_test, y_pred)
        metrics = {}
        metrics["Mean Squared Error"] = mse
        metrics["Root Mean Squared Error"] = rmse
        metrics["R^2 Score"] = r2_scr
        logger.debug("Model evaluated successfully")
        return metrics
    except Exception as e:
        logger.debug("Unexpected error occurred while evaluating the model: %s", e)
        raise
    
def save_metrics(evaluation_metrics: dict, metrics_dir: str) -> None:
    try:
        os.makedirs(os.path.dirname(metrics_dir), exist_ok = True)
        with open(metrics_dir, 'w') as file:
            json.dump(evaluation_metrics, file, indent=4)
        logger.debug("Metrics saved successfuly to %s", metrics_dir)
    except  Exception as e:
        logger.debug("Unexpected error occurred while saving the metrics: %s", e)
        raise





def main():
    try:
        test_path = "data/Processed_data/processed_test.csv"
        model_dir = 'models'
        
        modeltype = "lasso"
        
        if modeltype == "linear":
            filename = 'lr_model.pkl' 
        elif modeltype == "ridge":
            filename = 'ridge_model.pkl'
        elif modeltype == 'lasso':
            filename = 'lasso_model.pkl'
                     
        os.makedirs(os.path.join(model_dir, modeltype), exist_ok=True)
        X_test, y_test = load_data(test_path)
        model = joblib.load(os.path.join(os.path.join(model_dir, modeltype),filename))
        evaluation_metrics = evaluate_model(X_test, y_test, model)
        metrics_dir = 'reports'
        model_metric_dir = os.path.join(metrics_dir, modeltype)
        json_dir = os.path.join(model_metric_dir, 'metrics.json')
        save_metrics(evaluation_metrics, json_dir)
        
    except Exception as e:
        logger.debug("Unexpected error ccurred while evaluating the model: %s", e)
        raise
    
if __name__ == "__main__":
    main()