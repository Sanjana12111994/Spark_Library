import os
from pathlib import Path
path = Path(__file__).resolve().parents[0]
os.chdir(path)
import sys
from typing import List, Dict

from pyspark.sql import DataFrame

from EncodeSupervisedRatio import encode_supervised_ratio
from EncodeWeightedEvidence import encode_weighted_evidence



class FeatureEncoder(object):
    """ A general class for quick usage of Feature Encoding techniques
    General structure of code inspired by:
        https://github.com/apache/spark/blob/master/python/pyspark/ml/feature.py.
    """
    methods = ["SR", "WOE"]
    def __init__(self,
                 strategy: str = "SR",
                 cat_col: str= None,
                 label_col: str = None,
                 positive_class: str ='1',
                 negative_class : str='0',
                 bias: bool = False,
                
                ):
        """
        :param strategy         :  The selected strategy. Currently supported strategies are Supervised Ration Encoding (SR) 
                                   and Weight of Evidence (WOE).
        :param cat_col          :  The String that denotes the categorical column name.
        :param label_col        :  The String that denotes the label/target column name.
        :param positive_class   :  The String value in the label_col to denote positive class label. 
                                   default value = '1'
        :param negative_class   :  The String value in the label_col to denote negative class label.
                                   default value = '0'
        :bias                   :  A flag variable to indicate whether or not to include bias while encoding feature
                                   default value = False
     
        """
        self.strategy = None
        self.cat_col = None
        self.label_col = None
        self.positive_class = None
        self.negative_class = None
        self.bias = None
        
        
        self._setup(strategy, cat_col, label_col,positive_class,negative_class,bias)

    def _setup(self, strategy: str,
               cat_col: str,
               label_col: str,
               positive_class: str,
               negative_class: str,
               bias: bool,
               
               ) -> None:
        """Sets up attributes in a way that invalid inputs are caught at initialization."""

       

        self.set_strategy(strategy)
        self.set_cat_col(cat_col)
        self.set_label_col(label_col)
        self.set_positive_class(positive_class)
        self.set_negative_class(negative_class)
        self.set_bias(bias)
       
        
    def encode(self, df: DataFrame) -> DataFrame:
        strategy = self.get_strategy().upper()  # handle case where strategy is not uppercase
        strategy_mappings = {
            "SR": encode_supervised_ratio,
            "WOE": encode_weighted_evidence,
        }

      

        encode_method = strategy_mappings[strategy]
        if strategy in ["SR", "WOE"]:
            encoded_df = encode_method(df=df,
                                   cat_col=self.cat_col,
                                   label_col=self.label_col,
                                   positive_class=self.positive_class,
                                   negative_class=self.negative_class,
                                   bias=self.bias,
                                   )

            return encoded_df

    # # # # # # # # # # # # # # # #
    # # #  GETTERS & SETTERS  # # #
    # # # # # # # # # # # # # # # #

    def set_strategy(self, new_strategy: str) -> None:
        valid_strategies = ("SR", "WOE")

        if type(new_strategy) != str:
            raise Exception("Strategy must be a string.")
        elif new_strategy.upper() not in valid_strategies:
            raise Exception("""Strategy: {} is not supported. Current valid strategies are:
                               'SR', 'WOE'."""
                            .format(new_strategy))
        self.strategy = new_strategy.upper()

    def set_cat_col(self, new_cat_col: str) -> None:
        if type(new_cat_col) != str:
            raise Exception("Category column name must be a string.")
        elif len(new_cat_col) == 0:
            raise Exception("Category column name cannot be of length 0.")

        self.cat_col = new_cat_col

    def set_label_col(self, new_label_col: str) -> None:
        if type(new_label_col) != str:
            raise Exception("Label column name must be a string.")
        elif len(new_label_col) == 0:
            raise Exception("Label columns cannot be of length 0.")

        self.label_col = new_label_col

    def set_positive_class(self, new_positive_class: str) -> None:
        self.positive_class = new_positive_class
    
    def set_negative_class(self, new_negative_class: str) -> None:
        self.negative_class = new_negative_class

    def set_bias(self, new_bias: bool) -> None:
        if type(new_bias) != bool:
             raise Exception("\n The bias argument must be of type boolean \n Set bias = True to include bias in your computation \
                 \n The default setting is bias = False.")
        self.bias = new_bias
 

    def get_strategy(self) -> str:
        return self.strategy

    def get_cat_col(self) -> str:
        return self.cat_col

    def get_label_col(self) -> str:
        return self.label_col
    
    def get_positive_class(self) -> str:
        return self.positive_class

    def get_negative_class(self) -> str:
        return self.negative_class
    
    def get_bias(self) -> bool:
        return self.bias 

   