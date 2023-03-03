import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import traceback
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datetime import datetime
from PIL import Image

#local imports
from datasets_cifar10 import train_dataset, test_dataset
from model import Net
FP_PATH = "./results/false_positives/"


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
        
def compute_calibration_metrics(model, dataLoader, labelClasses, K=10):
    ''' computes the calibration metrics for the loaded model against the chosen dataset.
        model's input shape and output shape needs to match provided dataset input shape and target set respectively.

    
    '''
    # Set model to evaluation mode
    model.eval()

    # Initialize variables to store calibration metrics
    num_samples = 0
    confidences = []
    accuracies = []
    num_classes= len(labelClasses)
    resultsDict = {}
    results = []
    num_samples = 0
    confidences = []
    accuracies = []
    outputs = []
    results = []
    probability=[]
    labels = []
    # Iterate over the data loader and collect predictions and ground truth
    for inputs, targets in dataLoader:
    # for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Make predictions
        outputs = model(inputs)
        predictions = F.softmax(outputs, dim=1).detach().cpu().numpy()
        results.extend(predictions)
        probability.extend(np.amax(predictions, axis=1))
        # Update calibration metrics
        num_samples += len(targets)
        confidences.extend(np.amax(predictions,axis=1))
        labels.extend(targets.cpu().numpy())
        accuracies.extend(targets.cpu().numpy() == np.argmax(predictions, axis=1))

    # Calculate expected calibration error
    ece = 0
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_calibration = []

    bin_boundaries = np.linspace(0, 1, K+1)
    for bin_idx in range(K):
        in_bin = np.logical_and(confidences >= bin_boundaries[bin_idx], confidences < bin_boundaries[bin_idx+1])

        if np.sum(in_bin) > 0:
            bin_cal = np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin]))
            bin_calibration.append( bin_cal)
            ece += bin_cal
        else:
            bin_calibration.append(0)
    ece *= 100
    
    sn.set_theme(style="whitegrid")
    # print(bin_calibration)
    plt.figure(figsize = (10,7))
    for i , value in enumerate(bin_calibration):
        plt.text(labelClasses[i], value+(value*0.01), "{:.2f}%".format(value*100), horizontalalignment='center',     verticalalignment='center')
    graph_data= pd.DataFrame([bin_calibration],columns=[labelClasses])
    plt.bar(labelClasses,bin_calibration,)
    plt.title("Calibration error per class")
    plt.xticks(labelClasses)
    plt.ylabel("Error %")
    plt.xlabel("Class ID")
    graph_name = "./results/{}_{}_calibration_graph.png".format(checkpoint.__class__.__name__,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    plt.savefig(graph_name)

    

    # Calculate max calibration error
    mce = 0
    for bin_idx in range(K):
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
        
        labelClasses= [0,1,2,3,4,5,6,7,8,9]
        
        #   make sure the directory for the FP images exists
        for  i in labelClasses:
            if not os.path.exists("{}{}".format(FP_PATH,i)):
                os.mkdir("{}{}".format(FP_PATH,i))

        import torchvision.transforms as T
        from PIL import Image
        transform = T.ToPILImage()
        # Iterate over the data loader and collect predictions and ground truth
        FP_count = 0
        for i, (inputs, targets) in enumerate(dataLoader):
            # if i >0: break
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
            for j, (target) in enumerate(targets):
                predictedLabel = np.argmax(predictions[j])
                if target.cpu() != predictedLabel:
                    FP_count += 1
                    n = inputs[j].cpu().numpy()
                    img = Image.fromarray(((n.transpose(1,2,0)*0.5+0.5)*255).astype(np.uint8))

                    img.save("{}{}/FP_{}_ActualLabel_{}.jpg".format(FP_PATH,predictedLabel,FP_count,target.cpu()))

        print("") ## newline for output formatting
        


        # build_classification_report(labels,results, labelClasses)

        # build_confusion_matrix(labels,results, labelClasses)
    
        expected_calibration_error, max_calibration_error = compute_calibration_metrics(model, dataLoader,labelClasses)
        print("Expected Calibration Error: {:.2f}%".format(expected_calibration_error))
        print("Max Calibration Error: {:.2f}%".format(max_calibration_error))


        #save false positive to folder
        



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

