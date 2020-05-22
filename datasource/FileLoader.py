# -*- coding: utf-8 -*-
from datasource.Loader import Loader #you  must import the abstract class
from scipy.io import arff
import pandas as pd
import logging as log

#logging.config.fileConfig("log_config.ini", disable_existing_loggers=False)

class FileLoader(Loader):
    format =""
    #log.basicConfig(level=logging.DEBUG)
    def __init__(self,format,path):
        self.format= format
        super().__init__(path)
        print("File Loader enabled")
        
        
    def load(self):
        print (self.format)
        try:
            df= None
            if self.format == 'csv':
                df=pd.read_csv(self.path, sep=';')
            elif self.format== 'arff':
                df=pd.DataFrame(arff.loadarff(self.path))
            log.info("File Loaded")
            return df
        except FileNotFoundError:
           print("The file does not exist")
            
