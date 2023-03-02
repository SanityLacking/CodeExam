import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
import numpy as np
import sys
import traceback
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datetime import datetime

#local imports
from datasets_cifar10 import train_dataset, test_dataset
from model import Net

def build_classification_report(labels, predictions, labelClasses=[],Print=True):
    ''''
        build the sklearn classification report for the model's inference.
    '''
    target_names = []
    for c in labelClasses:
        target_names.append("Class {}".format(c))
    report = classification_report(labels, predictions, target_names=target_names)
    if Print:
        print('\nClassification Report\n')
        print(report)
    return report
    

def build_confusion_matrix(labels, predictions, labelClasses=[],Print=True):
    ''' builds a confusion matrix showing the interaction of the different classes, and distribution of TP and FP between each class.
        model's input shape and output shape needs to match provided dataset input shape and target set respectively.
        produces a matplotlib graph output and saves it to the results folder.
    '''
    
    df_confusion = pd.crosstab([predictions], [labels], rownames=['Actual Class'], colnames=['Predicted Class'], margins=True)
    print(df_confusion)
    

    plt.figure(figsize = (10,7))
    sn.heatmap(df_confusion, annot=True,cbar=False,cmap="plasma_r",fmt='d',linewidth=.5)    
    mapName = "./results/{}_{}_confusionMatrix.jpg".format(checkpoint.__class__.__name__,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    plt.savefig(mapName)
    print("HeatMap saved to results folder as: {}".format(mapName))    

    return True
        
def compute_calibration_metrics(model, dataLoader):
    ''' computes the calibration metrics for the loaded model against the chosen dataset.
        model's input shape and output shape needs to match provided dataset input shape and target set respectively.

    
    '''
    # Set model to evaluation mode
    model.eval()

    # Initialize variables to store calibration metrics
    num_samples = 0
    confidences = []
    accuracies = []

    # Iterate over the data loader and collect predictions and ground truth
    for i, (inputs, targets) in enumerate(dataLoader):
    # for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Make predictions
        outputs = model(inputs)
        predictions = F.softmax(outputs, dim=1).detach().cpu().numpy()

        # Update calibration metrics
        num_samples += len(targets)
        confidences.extend(predictions[:, 1])
        accuracies.extend(targets.cpu().numpy() == np.argmax(predictions, axis=1))

    # Calculate expected calibration error
    ece = 0
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, 11)
    for bin_idx in range(10):
        in_bin = np.logical_and(confidences >= bin_boundaries[bin_idx], confidences < bin_boundaries[bin_idx+1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin]))
    ece *= 100

    # Calculate max calibration error
    mce = 0
    for bin_idx in range(10):
        in_bin = np.logical_and(confidences >= bin_boundaries[bin_idx], confidences < bin_boundaries[bin_idx+1])
        if np.sum(in_bin) > 0:
            mce = max(mce, np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * 100)

    # Return calibration metrics
    return ece / 10, mce

 

def evalModel(model):
    try:
        #load the dataset and set the batchsize
        batch_size = 32        
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        traceback.print_exc()
        print("the dataset was not able to load, please check the exception message")
    try:
            
        resultsDict = {}
        predictions = []
        results = []
        num_samples = 0
        confidences = []
        accuracies = []
        outputs = []
        results = []
        labels = []
        
        # Iterate over the data loader and collect predictions and ground truth
        for i, (inputs, targets) in enumerate(dataLoader):
            print("\rtest evaluation: "+str(i)+" of "+str(len(dataLoader)-1),end='')
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Make predictions
            outputs = model(inputs)
            predictions = F.softmax(outputs, dim=1).detach().cpu().numpy()

            # Update calibration metrics
            num_samples += len(targets)
            confidences.extend(predictions[:, 1])
            results.extend(np.argmax(predictions, axis=1))
            labels.extend(targets.cpu().numpy())
            accuracies.extend(targets.cpu().numpy() == np.argmax(predictions, axis=1))
        print("") ## newline for output formatting
        

        # df = pd.DataFrame(list(zip(results, labels,confidences, accuracies)), columns=["prediction","label","confidence","accuracy"])
        # print("Output DataFrame head")
        # print(df.head())

        labelClasses= [0,1,2,3,4,5,6,7,8,9]

        # build_classification_report(labels,results, labelClasses)

        # build_confusion_matrix(labels,results, labelClasses)
    
        expected_calibration_error, max_calibration_error = compute_calibration_metrics(model, dataLoader)
        print("Expected Calibration Error: {:.2f}%".format(expected_calibration_error))
        print("Max Calibration Error: {:.2f}%".format(max_calibration_error))

    except Exception as e:        
        traceback.print_exc()
        print("model was not able to run, please check the exception message:")
    return True


# check if the module is called as an entry point or not.
if __name__ == '__main__':
    #check for command line arguments
    if len(sys.argv) > 1:
        modelName = sys.argv[1]
        print("loading model stored at: '", modelName +"'")
        #load the model and run evaluation code
        try:
            checkpoint = torch.load(modelName)   
            if isinstance(checkpoint, torch.jit.ScriptModule):
                print("warning, the supplied model has been loaded as a torchscript model, take caution")
            device = torch.device('cuda:0')
            checkpoint = checkpoint.to(device)
            evalModel(checkpoint)

        except Exception  as e:
            traceback.print_exc()
            print("could not load torch model file, please check error msg above")

            
    else:
        print("No modelPath argument provided.")

