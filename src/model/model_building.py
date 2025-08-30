import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from xgboost import XGBClassifier


# Logging configuration
try:
    logger = logging.getLogger('model_training')
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('model_training.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error configuring logging: {str(e)}")
    raise

# load parameters from params.yaml

def load_params():
    try:
        logger.info("Loading model parameters")
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)
        model_params = params['model_building']
        logger.info("Model parameters loaded successfully")
        return model_params
    
    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        raise

# load the train data
def load_training_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info("Loading training data")
        train_data = pd.read_csv(file_path)
        
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values
        
        logger.debug(f"Data loaded successfully. Shape: {X_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        logger.error("Training data file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

# train the model
def train_model(X_train, y_train, model_params):
    try:
        logger.info("Training the XGBClassifier model")
        model = XGBClassifier(
            reg_lambda=model_params['reg_lambda'],
            reg_alpha=model_params['reg_alpha'],
            colsample_bytree=model_params['colsample_bytree'],
            subsample=model_params['subsample'],
            min_child_weight=model_params['min_child_weight'],
            max_depth=model_params['max_depth'],
            learning_rate=model_params['learning_rate'],
            n_estimators=model_params['n_estimators'],
            gamma=model_params['gamma']
        )
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def save_model(model):
    try:
        logger.info("Saving trained model")
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    try:
        # Load parameters
        model_params = load_params()
        
        # Load training data
        X_train, y_train = load_training_data('./data/processed/train_scaled.csv')
        
        # Train model
        model = train_model(X_train, y_train, model_params)
        
        # Save model
        save_model(model)
        
        logger.info("Model building pipeline completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in model building pipeline: {str(e)}")
        raise
    finally:
        logger.info("Model building process finished")

if __name__ == "__main__":
    main()