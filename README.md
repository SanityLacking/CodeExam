

Setup: <br>

    1. Download the data <br>
        Cifar10: <br>
        || python datasets_cifar10.py
    2. Prepare the model <br>
        either use your own model, or run <i>model.py</i> to create a simple test model
        
Run the Script: <br>
    || python test.py





limitations: <hr>

    Because of the nature of pytorch models, the model's architecture needs to be known and available as a constructor so that it the model can be loaded later on, this means that either the evaluation script needs to be run in an environment that already knows the model's architecture, or the architecture needs to be passed alongside the model to the new enviornment to enable the evaluation of the model.
<br>
        This is more a limitation of pytorch and its model storage techniques, to solve this an alternative saving method could be used, such as converting the model to onyx, using torch.jit, etc. but these require additional work and come with their own limitations, especially when making custom tensor structures.
<br>

    For the purposes of this exam, I stuck with three options, either the model is loaded as a statedict, the entire model, or as a torchscript file. For the first two options it is assumed that the models structure is available to the python environment. 