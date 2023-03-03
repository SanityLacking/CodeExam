# CodeExam
ML ops CodeExam 


## Setup: <br>

    1. Prepare the model <br>
        either use your own model, or run <i>model.py</i> to create a simple test model and <i> train.py</i> to train it.
        
Run the Script: <br>
    
    python test.py "path/to/model.pt" --data "path/to/data"



### Outputs: <hr>
The script provides a selection of different metrics and reports on the model's performance including:
1. Classification Report per class consisting of precision, recall, f1-score for each individual class.
2. Class confusion matrix for each class, this plot is saved to <i> ./results/ </i>
3. The model's Expected Calibration Error 
4. The individual Class Calibration Error
5. The model's Maximimum Calibration Error
6. A Class knee chart for accuracy to confidence trade off for each class.

Additionally False positive inputs are identified and saved in <i> ./results/false_positives/</i> according to predicted label.

### Limitations: <hr>

#### Model loading
Because of the nature of pytorch models, the model's architecture needs to be known and available as a constructor so that it the model can be loaded later on, this means that either the evaluation script needs to be run in an environment that already knows the model's architecture, or the architecture needs to be passed alongside the model to the new enviornment to enable the evaluation of the model.
<br>
This is more a limitation of pytorch and its model storage techniques, to solve this an alternative saving method could be used, such as converting the model to onyx, using torch.jit, etc. but these require additional work and come with their own limitations, especially when making custom tensor structures.
<br>
For the purposes of this exam, I stuck with three options, either the model is loaded as a statedict, the entire model, or as a torchscript file. For the first two options it is assumed that the models structure is available to the python environment. 

#### False Positives
I was not able to fully complete the suggested unsupervised approach upon the false positives to find patterns, instead I added a knee test functionality for each class to illustrate the relationship of accuracy to model confidence for each class. This too could be improved to focus on class entropy rather then class throughput count. 

