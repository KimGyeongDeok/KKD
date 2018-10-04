# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:38:36 2018

@author: kkd
"""

import os
from numpy import array

def run(FOLDER_DIR,number_dims=36):
    train_or_test=['train','test']
    for i in train_or_test:
        FOLDER_DIR = FOLDER_DIR+'\\'+i
        folder_list = array(os.listdir(FOLDER_DIR))
        
        for index in range(len(folder_list)):
            path = os.path.join(FOLDER_DIR, folder_list[index])
            path_to_exe = path
            path = path + '/'
            off_list = os.listdir(path)
            
            for index in range(len(off_list)): # unneccesary file(binvox.exe) delete
                if off_list[index] =="binvox.exe":
                    index_to_delete=index
                    
            off_list.remove(off_list[index_to_delete])
                    
            for off in off_list:
                os.chdir(path_to_exe)
                os.system("binvox.exe -d "+str(number_dims)+" "+off)