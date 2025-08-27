import os
import sys
import numpy as np
import pandas as pd
from src.utils import evalute_model, save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    model_obj_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_path=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                # "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            params = {
                "Linear Regression": {},
                
                "Lasso": {
                    'selection': ['cyclic', 'random'],
                    'max_iter': [50, 100],
                    'alpha': [0, 0.1, 0.3, 1]
                },

                "Ridge": {
                    'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'sag', 'saga'],
                    'max_iter': [50, 100],
                    'alpha': [0, 0.1, 1, 0.5]
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 2, 3, 6],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'splitter': ['best', 'random']
                },

                "Random Forest Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None]
                },

                "AdaBoost Regressor": {
                    'n_estimators': [20, 50, 100],
                    'learning_rate': [0.5, 1.0, 0.3],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report:dict=evalute_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score=max(sorted(list(model_report.values())))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info('Best model found on both training and testing datasets')


            save_object(
                file_path=self.model_path.model_obj_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            score=r2_score(y_test,predicted)

            return score
        



        except Exception as e:
            raise CustomException(e,sys)
            
