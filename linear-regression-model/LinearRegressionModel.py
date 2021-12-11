# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:37:59 2020

@author: Ramachandran R

@Ref: https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/
"""

# Linear Regression With Stochastic Gradient Descent for Code (Bugs) Quality
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
from math import log
import math
from scipy.stats import mannwhitneyu
import csv
import xlsxwriter
from sklearn import metrics

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file,delimiter=',')
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
	
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])-1):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
	
# Find the information Gain for every fetures
def infoG(x,y):
    mx=np.mean(x)
    inl=np.where(x<=mx)
    inh=np.where(x>mx)
    yl=y[inl[0]]
    yh=y[inh[0]]
    nl0=len(np.where(yl==0)[0])
    nl1=len(np.where(yl>0)[0])
    lchild=[nl0, nl1]
    enl=entropy(lchild)
    nh0=len(np.where(yh==0)[0])
    nh1=len(np.where(yh>0)[0])
    rchild=[nh0, nh1]
    enr=entropy(rchild)
    low=nl0+nh0
    high=nl1+nh1
    total=len(y)
    par=[(low),(high)]
    enp=entropy(par)
    infoGain=enp-(len(yh)/total)*enr-(len(yl)/total)*enl
    
    return infoGain

#return the Entropy of a probability distribution
def entropy(pi):
    total = 0
    for p in pi:
        if sum(pi)==0:
            p = p / (sum(pi) + 1.0)
        else:
            p = p / sum(pi)
        if p != 0:
            total += p * log(p, 2)
        else:
            total += 0
    total *= -1
    return total

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			if (minmax[i][1] - minmax[i][0]) == 0:
				row[i] = (row[i] - minmax[i][0]) / ((minmax[i][1] - minmax[i][0]) + 1.0)
			else:
				row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	accuracies = list()
	fM = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		coef,predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		Y_act=np.array(actual)
		Y_act[np.where(Y_act>0)]=1
		Y_Pre=np.array(predicted)
		Y_Pre[np.where(Y_Pre>0)]=1
		Y_Pre[np.where(Y_Pre<=0)]=0
		rmse = rmse_metric(Y_act, Y_Pre)
		accuracy = callAcc(Y_act, Y_Pre)
        
		pr = performance(Y_act, Y_Pre)
        
		scores.append(rmse)
		accuracies.append(accuracy)
        
		fM.append(pr)
	return coef,max(scores),max(accuracies),max(fM)
#return coef,max(scores),max(accuracies),max(fM)

#Calculate Accuracy in percentage
def callAcc(yact, ypre):
    total= len(yact)
    correct = 0
    for i in range(len(yact)):
        if yact[i] == ypre[i]:
            correct += 1
    return (correct)/total

#Prepare confusion metric data
def conf_metric(yact, ypre):
    TP,TN,FP,FN = 0,0,0,0
    if yact == 1:
        TP = 1 if ypre == 1.0 else 0
        FN = 1 if ypre == 0.0 else 0
    if yact == 0:
        FP = 1 if ypre == 1.0 else 0
        TN = 1 if ypre == 0.0 else 0
    return TP,TN,FP,FN

#calculate the F-measure
def performance(yact, ypre):
    TP,TN,FP,FN = 0,0,0,0
    for i in range(len(yact)):
        tp,tn,fp,fn =conf_metric(yact[i], ypre[i])
        TP += tp
        TN += tn
        FP += fp
        FN += fn
    precision = 0 if (TP + FP) == 0 else (TP / (TP + FP))
    recall = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    return 0 if (precision + recall) == 0 else (2 * (precision * recall) / (precision + recall))

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat
 
# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	print("Inside coefficient SGD")
	for epoch in range(n_epoch):
		if (epoch%100 == 0):
			print(epoch)
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
	return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(coef,predictions)	

# load and prepare data
infoGain=[]
store=[]
for j in range(0, 56):
    print(j+1)
    filename='C:/Users/Ramji/BITS/SEMESTER_2/Data Mining (ISZC415)/Assignment/data/'+str(j+1)+'.csv'
    df=np.genfromtxt(filename,delimiter=',')
    features=df[:,0:-1]
    bugs=df[:,-1]
    '''
    F_infoGain=[]
    for i in range(0,20):
        F_infoGain.append(infoG(features[:,i],bugs))
    F_infoGain.insert(0, "proj_" + str(j+1))
    infoGain.append(F_infoGain)
    '''
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.0001
    n_epoch = 1000
    coef,scores,acc,fM = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
    coef.insert(0, "Project_" + str(j+1))
    coef.append(scores)
    coef.append(acc)
    coef.append(fM)
    store.append(coef)
    print('Completed: ' + str(j+1) + '.csv file')

Output = 'C:/Users/Ramji/BITS/SEMESTER_2/Data Mining (ISZC415)/Assignment/data/Sample.xlsx'
    
workbook   = xlsxwriter.Workbook(Output)

#worksheet1 = workbook.add_worksheet('Information Gain')
worksheet2 = workbook.add_worksheet('LinearRegressionModel')
'''
#headers
info_gain_header = ["Projects"]
for i in range(len(infoGain[0])-1):
    info_gain_header.append("Feature " + str(i + 1))
infoGain.insert(0, info_gain_header)
''' 
gradient_header = ["Projects"]
for i in range(len(store[0])-5):
    gradient_header.append("Coefficient " + str(i + 1))
gradient_header.append("Constant")    
gradient_header.append("RMSE")    
gradient_header.append("Accuracy")
gradient_header.append("F1 Score")

store.insert(0, gradient_header)

'''
#Write data in excel sheet
col = 0
for row, data in enumerate(infoGain):
    worksheet1.write_row(row, col, data)
'''
col = 0
for row, data in enumerate(store):
    worksheet2.write_row(row, col, data)

workbook.close()
