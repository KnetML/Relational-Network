#### A simple neural network module for relational reasoning

Knet implementation of Relational Networks - A simple neural network module for relational reasoning

Implemented & tested on CLEVR from state descriptions.

### Usage
* Download clever data (without images) and preprocess the data.
```
cd relational-network
bash prepare_data.sh
```
* Run experiment script to train the model on state descriptions
```
bash run.sh
```
* The model and log file will be generated inside `src/saved_models`


### Results

| Epoch        | Accuracy (Val Set)|
| ------------- |:-------------:|
| 1            | 44.07%         |
| 5            | 47.50%         |
| 15           | 57.69%         |
| 25           | 79.60%         |
| 40           | 93.21%         |
| 65           | 94.50%         |

*
![img1](/util/chart.png)


* Training Time: On Tesla K80 GPU, 1 epoch takes ~ 12 minutes.
