import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion.py')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

log_path = os.path.join(log_dir, 'data_ingestion.log')
fileHandler = logging.FileHandler(log_path)
fileHandler.setLevel('DEBUG')

Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(Formatter)
fileHandler.setFormatter(Formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        dataset = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return dataset
    
    except pd.errors.ParseError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnecessary columns and perform some basic processing"""
    try:
        df.drop(columns = ['Unnamed: 0', 'Name', 'Model'], inplace = True)
        brand_counts = df['Brand'].value_counts()
        # Get the list of brands to be replaced
        brands_to_replace = brand_counts[brand_counts < 9].index
        # Use .loc to find rows with these brands and replace them with 'Other'
        df.loc[df['Brand'].isin(brands_to_replace), 'Brand'] = 'Other'
        logger.debug("Data preprocessing completed")
        return df
        
    except KeyError as e:
        logger.debug("Missing column in the dataframe: %s", e)
        raise        
    except Exception as e:
        logger.debug("Unexpected error occurred while preprocessing data: %s", e)
        raise
    
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok = True)
        train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index = False)
        test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index = False)
        logger.debug("Train and test data saved to %s", raw_data_path)
    
    except Exception as e:
        logger.debug("Unexpected error occurred while saving the data: %s", e)
        raise
    
    
def main():
    try:
        test_size = 0.2
        data_path = "https://raw.githubusercontent.com/Aayoosss/Datasets/refs/heads/main/ndtv_data_final.csv"
        dataset = load_data(data_path)
        processed_data = preprocess_data(dataset)
        train_data, test_data = train_test_split(processed_data, test_size = test_size, random_state = 42, stratify=processed_data['Brand'])
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.debug("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()
    
    