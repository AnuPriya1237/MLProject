import sys
import pandas as pd
import numpy as np
import os
from  src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_tranforation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Get the numerical and categorical pipeline ready
        '''
        try:
            logging.info('fetching training data from artifacts')
            data=pd.read_csv('artifacts/data.csv')
            target_column='math_score'
            independent_feature=data.drop(columns=[target_column])
            logging.info("seperating numerical and categorical features...")
            num_feature = independent_feature.select_dtypes(exclude=object).columns
            categorical_feature= independent_feature.select_dtypes(include=object).columns

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler()),
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('oneHotEncoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False)),

                ]
            )

            logging.info(f'Numerical columns standard scaling completed: {num_feature}')
            logging.info(f'Categorical columns encoding completed: {categorical_feature}')

            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,num_feature),
                 ('cat_pipeline',cat_pipeline,categorical_feature)
                 ],
                remainder='passthrough'
                
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transforation(self,train_path,test_path):
        logging.info("Transforming data...")
        try:
            logging.info("fetching train and test data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info('Obtaining preprocessor object')
            preprocessing_object=self.get_data_transformer_object()
            target_column='math_score'

            logging.info('train and test df with independent nad target features')
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                f" Applying preprocessor object on training and train  dataframe"
            )
            print(input_feature_test_df.head())
            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f'saved preprocessing object')
            save_object(
                self.data_tranforation_config.preprocessor_obj_file_path,
                preprocessing_object
            )

            return (
                train_arr,
                test_arr,
                self.data_tranforation_config.preprocessor_obj_file_path
            )




        except Exception as e:
             raise CustomException(e,sys)


