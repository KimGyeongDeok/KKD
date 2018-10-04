# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:04:01 2018

@author: kkd
"""

import os
import numpy as np
import binvox_rw
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def return_input(binvox_file_list,numbering,DIR,folder_list):
    init=0
    for i in range(len(numbering)):
        for j in range(init,numbering[i]):
            array_path = os.path.join(DIR+'\\'+folder_list[i], binvox_file_list[j])
            with open(array_path,"rb") as f:
                temp=binvox_rw.read_as_3d_array(f)
            yield [np.array(temp.data)]
        init=numbering[i]
        
def return_label(numbering,onehot_encoded):
    init=0
    for i in range(len(numbering)):
        for j in range(init,numbering[i]):
            yield onehot_encoded[i]
        init=numbering[i]
        
def return_dims(binvox_file_list,numbering,DIR,folder_list):
    init=0
    for i in range(len(numbering)):
        for j in range(init,numbering[i]):
            array_path = os.path.join(DIR+'\\'+folder_list[i], binvox_file_list[j])
            with open(array_path,"rb") as f:
                temp=binvox_rw.read_as_3d_array(f)
            return temp.dims
        init=numbering[i]
        break

        
        


        
def make_examples_from_binvox(train_or_test): #just put either 'train' or 'text' in this def
    DIR = 'C:\\Users\\kkd\Desktop\\revised_folder_order_practical\\'+train_or_test
    folder_list = array(os.listdir(DIR))
    
    #train input, label declare
    input_set = []
    label_set = []
    input_dims=0
    
    # LabelEncoder Class call and make onehot vectors
    label_encoder = LabelEncoder()  
    integer_encoded = label_encoder.fit_transform(folder_list)
    onehot_encoder = OneHotEncoder(sparse=False) 
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    numbering=[]
    for index in range(len(folder_list)):
        path = os.path.join(DIR, folder_list[index])
        path = path + '/'
        all_file_list = os.listdir(path)
        if index==0:
            binvox_file_list=[]
        
        #classify 
        for index_delete in range(len(all_file_list)): # unneccesary file(binvox.exe) delete
            if all_file_list[index_delete] =="binvox.exe":
                index_to_delete=index_delete
        
        all_file_list.remove(all_file_list[index_to_delete])
        
        binvox_indices=[]
        for index_classify in range(len(all_file_list)):
            fname, ext = os.path.splitext(all_file_list[index_classify])
            if ext == ".binvox":
                 binvox_indices.append(index_classify)
        
        for index_add in range(len(binvox_indices)):
            add=all_file_list[binvox_indices[index_add]]
            binvox_file_list.append(add)
        numbering.append(len(binvox_file_list))
        
    input_set=return_input(binvox_file_list,numbering,DIR,folder_list)
    label_set=return_label(numbering,onehot_encoded)
    input_dims=return_dims(binvox_file_list,numbering,DIR,folder_list)
    number_all_of_files=len(binvox_file_list)
    number_of_class=len(folder_list)
    value=[input_set,label_set,input_dims,number_all_of_files,number_of_class]
    
    return value

#j=0
#a= make_examples_from_binvox('test')
#train_input=a[0]
#train_label=a[1]
#dims=a[2]
#n=a[3]
#c=a[4]