# Class-wise Shapley value-based Augmentation (CSA)
This is the official implementation for NeurIPS2022 paper: 

Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain

Yiting Chen, Qibing Ren and Junchi Yan

## Install Requirements
The codebase is built and tested with Python 3.9.5. To install required packages:

```pip install -r requrements.txt```

## Calculate the Shapley value
To sample Shapley value, the first step is to train a ST model via

 ```python train_model.py --config config/train_model.yaml```

The checkpoint of the trained model will be saved in ```./output/$date_of_the_time/train_model.yaml/params/```

Change the ```shapley.model_path``` in ```config/Shapley.yaml``` to the path of checkpoint at the last epoch and sample the Shapley value via.

```python Shapley_softmax.py --config config/Shapley.yaml```

The results would be saved in ```./output/$date_of_the_time/Shapley.yaml/shap_result```

## Train AT models with CSA
To train At models with CSA, the first step is get the NFCs and PFCs of each data sample:

```python Reconstruct.py --shap_path ./output/$date_of_the_time/Shapley.yaml```

where the reconstructed images of the NFCs and the PFCs of each data sample will be stored in ```./output/$date_of_the_time/Shapley.yaml/ifft```

You could also download the generated files [here](https://drive.google.com/file/d/1do8KbtySg7vCZr0cXCQHZ4m0HViIPfdR/view?usp=sharing)

Change the ```train.conf_path``` in ```config/madrys_CIFAR10_ResNet18_csa.yaml``` and ```config/trades_CIFAR10_ResNet18_csa.yaml``` to the path ```./output/$date_of_the_time/Shapley.yaml/ifft``` or where ever you place the reconstructed images.

Train a ResNet18 under PGD-AT with CSA:

```python train_model_adv_csa.py --config config/madrys_CIFAR10_ResNet18_csa.yaml```

Train a ResNet18 under TRADES with CSA:

```python train_model_adv_csa.py --config config/trades_CIFAR10_ResNet18_csa.yaml```

We also provide checkpoints of model at the last epoch:

| Methods    | Clean Acc | Acc under PGD-20 | Acc under Auto attack | Checkpoint                                                                                     |
|------------|-----------|------------------|-----------------------|------------------------------------------------------------------------------------------------|
| PGD-AT     | **84.49** | 46.38            | 44.06                 | [ResNet18](https://drive.google.com/file/d/1gi3EdttHumTqpwITEJiguQXLUFlOCZBe/view?usp=sharing) |
| PGD-AT+CSA | 82.91     | **49.42**        | **46.56**             | [ResNet18](https://drive.google.com/file/d/1_GCjLWAhDqxxjY2El-E9yw1gNdxfdJaM/view?usp=sharing) |
|            |           |                  |                       |                                                                                                |
| TRADES     | **81.71** | 51.08            | 47.74                 | [ResNet18](https://drive.google.com/file/d/14XaIwXK9lNH6eokzAs2BFigdSfezt2JX/view?usp=sharing) |
| TRADES+CSA | 81.62     | **52.06**        | **49.16**             | [ResNet18](https://drive.google.com/file/d/1W7BSAw2xnfCEizxjWkOJWhPRGsa-FVmC/view?usp=sharing) |


