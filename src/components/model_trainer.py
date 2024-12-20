import os 
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.utilis import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_path_file=os.path.join('artifacts',"model.pkl")
class ModelTrainer:
    def __init__(self) :
       self.model_trainer_config=ModelTrainerConfig()  
       
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test data input")
            X_train,y_train,X_test,y_test=(
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': [None, 'log2']
                },
                "Random Forest": {  # Corrected to match the model name
                    'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [32, 64, 128, 200, 256]
                },
                "Gradient Boosting": {
                    'criterion': ['friedman_mse', 'squared_error'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['log2', 'sqrt',None],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'loss': ['absolute_error', 'squared_error', 'huber', 'quantile'],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                },
                "XGBRegressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 200, 256],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                },
                "CatBoosting Regressor": {
                    'iterations': [30, 50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                },
                "Linear Regression": {},
                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 200, 256],
                    'loss': ['linear', 'exponential', 'square'],
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                }
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
             
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_path_file,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
            
        
             