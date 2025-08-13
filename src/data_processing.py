import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
import logging
import os

log_path = 'logs'

logger = logging.getLogger('data_processing.py')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

log_file_path = 'data_processing.log'
fileHandler = logging.FileHandler(os.path.join(log_path, log_file_path))
fileHandler.setLevel('DEBUG')

Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileHandler.setFormatter(Formatter)
consoleHandler.setFormatter(Formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def fitencoders(categorical_df: pd.DataFrame) -> dict:
    try:
        encoders = {}
        Ordinal_order = [['No', 'Yes']]
        Columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
        oe = OrdinalEncoder(categories=Ordinal_order * len(Columns), dtype = 'int64')
        oe.fit(categorical_df[Columns])
        encoders['ordinal'] = oe
        encoders['ordinal_cols'] = Columns
        logger.debug("Ordinal Encoder Fitted")
        
        Columns = ['Operating system', 'Brand']
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output = False, dtype = 'int64').set_output(transform='pandas')
        ohe.fit(categorical_df[Columns])
        encoders['onehot'] = ohe
        encoders['ohe_cols'] = Columns
        logger.debug("OneHotEncoder Fitted")
        
        return encoders
    
    except Exception as e:
        logger.debug("Unexpected error occured while fitting encoder: %s", e)
        raise
    
def fitscaler(numeric_df: pd.DataFrame) -> sklearn.preprocessing._data.MinMaxScaler:
    try:
        numeric_df = numeric_df.drop(columns = ['Price'])
        Scaler = MinMaxScaler()
        Scaler.fit(numeric_df)
        logger.debug("Scaler fitted successfully")
        return Scaler
    
    except Exception as e:
        logger.debug("Unexpected error occurred while fitting scaler: %s", e)
        raise
    
def transformcolumns(categorical_data_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    try:
        categorical_df = categorical_data_df.copy()
        categorical_df[encoders['ordinal_cols']] = encoders['ordinal'].transform(categorical_df[encoders['ordinal_cols']])
        logger.debug("Ordinal encoding successfull")
        transformed_ohe_data = encoders['onehot'].transform(categorical_df[encoders['ohe_cols']])
        categorical_df = pd.concat([categorical_df, transformed_ohe_data], axis = 1).drop(columns = encoders['ohe_cols'])
        logger.debug("One hot encoding successfull")
        return categorical_df
    
    except Exception as e:
        logger.debug("Unexpected error occurred while transforming columns: %s", e)
        raise


def scalecolumns(numeric_data_df: pd.DataFrame, Scaler: sklearn.preprocessing._data.MinMaxScaler) -> pd.DataFrame:
    try:
        numeric_df = numeric_data_df.copy().drop(columns = ['Price'])
        cols = numeric_df.columns
        scaled_data = Scaler.transform(numeric_df)
        Scaled_data = pd.DataFrame(scaled_data, columns=cols)
        Scaled_data = pd.concat([Scaled_data, numeric_data_df[['Price']]], axis = 1 )
        Scaled_data.head()
        logger.debug("Numeric columns scaled successfully")
        return Scaled_data
        
    except Exception as e:
        logger.debug("Unexpected error occured while scaling the numeric columns: %s", e)
        raise

def save_datasets(traindf: pd.DataFrame, testdf: pd.DataFrame, data_path: str):
    try:
        Processed_data_path = os.path.join(data_path, "Processed_data")
        os.makedirs(Processed_data_path, exist_ok = True)
        traindf.to_csv(os.path.join(Processed_data_path, 'processed_train.csv'), index=False)
        testdf.to_csv(os.path.join(Processed_data_path, 'processed_test.csv'), index = False)
        logger.debug("Dataset saved successfully")
        
    except Exception as e:
        logger.debug("Unexpected error occurred while saving the datasets: %s", e)
        raise
         



def main():
    try:
        train_path = 'data/raw/train.csv'
        test_path = 'data/raw/test.csv'
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        numeric_df = train_df[train_df.select_dtypes(include=['int64','float64']).columns]
        categorical_df = train_df[train_df.select_dtypes(exclude=['int64', 'float64']).columns]
        
        encoders = fitencoders(categorical_df)
        Scaler = fitscaler(numeric_df)
        
        
        categorical_df = transformcolumns(categorical_df, encoders)
        numerical_df = scalecolumns(numeric_df, Scaler)
        
        preprocessed_train_data = pd.concat([numerical_df, categorical_df], axis = 1)
        
        numeric_df = test_df[test_df.select_dtypes(include=['int64','float64']).columns]
        categorical_df = test_df[test_df.select_dtypes(exclude=['int64', 'float64']).columns]
        
        categorical_df = transformcolumns(categorical_df, encoders)
        numerical_df = scalecolumns(numeric_df, Scaler)
        
        preprocessed_test_data = pd.concat([numerical_df, categorical_df], axis = 1)
        data_path = 'data'
        save_datasets(preprocessed_train_data, preprocessed_test_data, data_path) 
        
    except Exception as e:
        logger.debug("Unexpected error occured while preprocessing: %s", e) 
        
        
if __name__ == "__main__":
    main()