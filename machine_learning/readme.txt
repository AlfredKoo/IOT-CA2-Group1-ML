main.ipynb contains the verbose output
the .py file are callable functions from another file used for the school assignment
---Notes---
The notebook contains a scrapped feature engineering attempt to alter the dataset used for training.
The idea is that if the car is stationary, it might also have a label of 1(dangerous) which can affect the model negatively.
Hence an attempt was made to average out the result, at first the accuracy looks high at 75%, however, i realised they were all false positives as
all the predictions returned were 0. Hence i bypasses that feature engineering part, but it is still in the code
The csv for the feature engineered dataset is stored in /datasets/newdataframe.csv
