# PIRSense-Activity-Detection-via-PIR-Sensor-Data

## Overview of the dataset
So, with the dataset there is also a Introductory research paper given, link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081223&tag=1

### Data Collection:
The traditional passive infrared sensor(PIR) finds difficulty in sensing both stationary and moving occupants, hence they used custom designed SLEEPIR sensor nodes. Each node includes traditional PIR sensors and custom SLEEPIR modules. Each collects one observation every 30 seconds. Data includes raw voltage signals, PIR output, and derived features over sliding time windows.  


The obtained data set is a time series dataset, that is the date samples are related and mainly in chronological order. Date and Time depicts when the data is collected. <br> <br>
Labels: depict 0: Vacancy; 1: Stationary human presence; 3: Other activity/motion respectively.<br> <br> 

Temperature depicts the ambient temperature(°F) of the environment. Sensors work/ sense the animals or humans based on the temperature produced by them. PIRs are passive devices i.e., they themselves don't produce temperature. Hence the dataset includes the temperature of the environment and objects only. The PIR sensor feature stores raw analog readings from the sensor. Each value represents IR activity (motion/heat) detected over a short 4-second window.


### Some Insights Derived from the Data
i. So, one thing we noticed is that a sensor is highly correlated with the sensor just adjacent to it, we also did some experiments with window sizes which is mentioned in detail further in this notebook <br>

ii. One Feature "Temperature_F" is very highly correlated with Label. <br>

iii. for a window size of 2 or more No feature is even moderately correlated with each other. <br>

iv. "Temperature_F" feature has only 6 unique values which is the least in the entire feature set. <br>

v. "Temperature_F" has the highest number of outliers among all the features. <br>

vi. The Dataset is not well balanced, like out of 7651 samples roughly 6247 samples are of class 0 itself

## Overview of the Project
1. Helper.py -> So we have created a python file in which we have added all the necessary processing, plotting functions as well as MLP and LSTM models.
2. Data Pre-processing <br>
So we decided to Create 3 more datasets from the full dataset <br> <br>

### A. Dataset Created by Manual Feature selection <br>
So We took these things into consideration for creating the manual datset <br>

i. Correlation Matrix <br>
ii. Heatmap <br>
iii. Variance Threshold <br>
iv. Information Gain <br>
v. Features with 0 Standard deviation <br>

Now after considering all the above mentioned principles we selected 20 features out of 56
### B. So now to check how good is our feature selection we decided to compare it with standard feature selection libraries, so picked two libraries PCA & Feature engine

So we created a data set with PCA which consists of 20 features which matches the number of features we have manualy selected

### C. Dataset created using the Feature Engine library. <br>
Feature engine is library which has inbuilt sub liraries like 

Missing data imputation
Encoding of categorical features
Discretization
Outlier capping or removal
Feature transformation
Creation of new features
Feature selection etc....

So from there Feature Selection library we used two models <br>
i. DropCorrelatedFeatures <br>
ii. SmartFeatureSelection <br>

Now the features selected by these 2 libraries are exactly same.
So we decided to create a dataset with these features

## 3. Models
When it comes to models we have chosen 2 Models <br>
i. MLP <br>
ii. LSTM <br>

Brief overview of the models <br>
i. MLP: <br>
    a. Architecure: Input Layer -> 64 -> 128 -> 64 -> 3 <br>
    b. Epochs: 75 <br> <br>
ii. LSTM<br>
    a. 64 Memory cells  -> 32 memory cells -> 3 (Dense) <br>
Batch size = 32 <br>
    b. Number of Timestep = 4 <br>

A detailed Explanation/ Justificaion of why we chose these architecures and other hyperparameters is given further in this notebook

## 4. Model performance
In brief <br>
MLP's are performing way better than LSTM's and also the time taken by MLP's is less compared to LSTM's <br>

So the evaluation function we chose MLP over LSTM 
Overall Analysis of MLP and LSTM Models Across Four Datasets <br>

The comparison between the MLP and LSTM models across four datasets—full dataset, manually selected features, PCA, and smart feature selection—highlights the consistent superiority of MLP in terms of accuracy, computational efficiency, stability, and class-specific performance. <br>

MLP achieves near-perfect accuracy (0.99) across all datasets and feature selection strategies, while LSTM struggles with lower accuracy, particularly under PCA (0.68), despite improving to 0.87 with smart feature selection. <br>

Precision and recall values further improves MLP's performance, as it maintains high scores across all classes, whereas LSTM faces significant challenges handling Class 1 effectively. <br> 
MLP is also far more computationally efficient, completing tasks in approximately 130 seconds across all datasets compared to LSTM's runtime exceeding 470 seconds. Additionally, MLP demonstrates exceptional stability with near-zero standard deviation across datasets, reflecting consistent predictions regardless of feature selection methods, while LSTM exhibits higher variability, especially under PCA (standard deviation of 0.36). <br>

In terms of F1 score, MLP delivers balanced and uniformly high scores for all classes and datasets, whereas LSTM performs well for Classes 0 and 3 but fails to handle Class 1 reliably. Although smart feature selection improves LSTM's performance, it still falls short compared to MLP's consistent excellence across all metrics and scenarios.
A detailed analysis of all the datasets and model evaluation is given in the end of the notebook

## Analysis of the models and the dataset

### Analysis of mean accuracy across datasets reveals that the: 

1. MLP consistently outperforms the Long Short-Term Memory (LSTM) model. 
2. On the full dataset, MLP achieves near-perfect accuracy (0.99), significantly higher than LSTM's 0.80. 
3. With manually selected features, MLP maintains its high accuracy (0.99), while LSTM shows slight improvement to 0.84 but still lags behind. 
4. When PCA is applied, MLP's accuracy drops slightly to 0.96, whereas LSTM's performance decreases further to 0.68, widening the gap between the two models. 
5. Finally, with smart feature selection, MLP again achieves its usual high accuracy (0.99), and LSTM improves to 0.87 but remains inferior to MLP in terms of accuracy across all datasets.

### Analysis of Computational Time:

1. LSTM model requires significantly more time for training and evaluation compared to the MLP. 
2. On the full dataset, LSTM takes 523.54 seconds, while MLP completes the task in just 133.92 seconds. 
3. Similarly, with manually selected features, LSTM takes 537.71 seconds, whereas MLP is much faster at 136.96 seconds. 
4. When PCA is applied, LSTM's runtime reduces to 469.51 seconds but remains far higher than MLP's efficient 132.70 seconds. 
5. Finally, with smart feature selection, LSTM takes 472.79 seconds compared to MLP's quick runtime of 129.25 seconds. Overall, MLP consistently demonstrates a significant advantage in computational efficiency across all scenarios.


### Analysis of F1-Score
1. Model Stability: <br>
    A.MLP exhibits consistent performance across all classes and datasets. <br>
    B. LSTM shows significant variability, particularly under PCA and for Class 1. <br>

2. Class-Specific Performance: <br>
    A. MLP achieves uniformly high F1 scores for all classes. <br>
    B. LSTM performs well for Classes 0 and 3 but fails to handle Class 1 effectively. <br>

3. Impact of Feature Selection Methods: <br>
    A. PCA negatively impacts both models but has a more severe effect on LSTM. <br>
    B. Smart Feature Selection improves LSTM's performance but still falls short compared to MLP.

### Analysis of Standard Deviation

1. MLP exhibits extremely low standard deviation across all datasets, indicating consistent and stable performance <br>
2. LSTM shows significantly higher standard deviation across all datasets, reflecting greater variability in its predictions <br>
3. MLP consistently has near-zero standard deviation, indicating reliable and stable predictions. <br>
4. LSTM's higher standard deviation points to inconsistent performance across datasets.<br>
5. PCA results in the highest variability for LSTM (0.36), while Smart Feature Selection reduces it to a more manageable level (0.07). <br>
6. MLP remains unaffected by feature selection methods, maintaining low variability throughout. <br>

### Analysis of Precision 
1. MLP consistently achieves high precision across all classes and feature selection methods, except for a slight drop in precision for Class 1 with PCA. <br>

2. LSTM struggles significantly with precision for Class 1 across all methods, but Smart Feature Selection improves its performance for this class. <br>

3. PCA negatively impacts both models, especially LSTM, where precision for Class 1 drops to near-zero (0.02). <br>

### Analysis of Recall values
1. MLP Consistently high recall values accross all feature selection methods <br>
2. PCA reduces the recall of Class 1 significantly compared to other methods <br>
3. Smart Feature Selection provides the best balance, achieving perfect recall for Classes (0 and 3) and near-perfect results for Class (1) <br>
4. Struggles with low recall for Class (1) across all methods.
5. PCA performs poorly, especially for Classes (0) and (1), while maintaining good results only for Class (3).

## Comparison of Datasets
1. The Full Dataset provides the highest recall and accuracy for both models, with MLP achieving near-perfect scores across all classes and LSTM performing well for Classes 0 and 3 but struggling significantly with Class 1. <br>

2. Manually Selected Features maintain MLP's strong performance with minimal variation, while LSTM shows slight improvement in recall for Class 1 but remains less effective overall. <br>

3. The PCA dataset results in the most significant performance degradation for both models, particularly for LSTM, where recall for Class 1 drops to near-zero. This indicates that PCA fails to preserve critical features necessary for distinguishing certain classes, especially for LSTM. <br>

4. Finally, Smart Feature Selection emerges as the most balanced approach, enabling MLP to maintain its high accuracy and recall while significantly improving LSTM's performance on Class 1 (recall rises to 0.73). However, even with Smart Feature Selection, LSTM's overall performance remains inferior to MLP across all metrics. 
Conclusion
MLP is the superior model overall due to its high accuracy, computational efficiency, stable predictions, and balanced class-wise performance across all datasets and feature selection methods. While LSTM can benefit from smart feature selection to improve its results slightly, it remains less reliable and slower than MLP. <br>

Smart Feature Selection emerges as the best one comparatively accross all datasets
