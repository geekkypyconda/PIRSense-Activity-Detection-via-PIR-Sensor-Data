import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import importlib
import os
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

class Processor():
    def __init__(self):
        pass

    def get_correlated_features(self, corr, window=0, Threshold = 0.5, print_it = False):
        high_corr_pairs = []
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                # Ensure col1 is not compared to itself, and nearby sensors are excluded
                if col1 != col2 and abs(i - j) > window and abs(corr.loc[col1, col2]) > Threshold:
                    high_corr_pairs.append((col1, col2))
        
        if print_it:
            print(f"Number of Highly Co-related features with window size {window}: {len(high_corr_pairs)}")

            for i in high_corr_pairs:
                print(i)

        return high_corr_pairs

    def get_correlatd_count(self, corr_pairs):
        cnt = {}
        for pair in corr_pairs:
            # check if it exists or not
            if pair[0] in cnt:
                cnt[pair[0]] = cnt[pair[0]] + 1
            else:
                cnt[pair[0]] = 1

        return cnt

    def separate_label(self, data):
        labels = data["Label"]
        data_without_labels = data.drop('Label', axis=1)

        return labels, data_without_labels

    def features_with_zero_std(self, data):
        std = data.std(numeric_only=True)
        zero_std_cols = std[std==0].index.tolist()

        return zero_std_cols

    def create_dataset(self, list_of_selected_features, data, separate_labels = False):
        selected_data = data[list_of_selected_features]

        if separate_labels == True:
            return self.separate_label(selected_data, True)
        else:
            return selected_data

    def scaler(self, data_without_labels, type = "min_max"):
        if type == 'min_max':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(data_without_labels)
            return X
        elif type == "standard":
            scaler = StandardScaler()
            X = scaler.fit_transform(data_without_labels)
            return X
        elif type == "quantile":
            scaler = QuantileTransformer()
            X = scaler.fit_transform(data_without_labels)
            return X
        else:
            print("Wrong Type of Scaler!")
            print("Aborting Operation")
            return None

    def one_hot_encoder(self, labels):
        unique_labels = sorted(labels.unique())

        # Create a dictionary mapping labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        # One-hot encode manually
        y_onehot = np.zeros((len(labels), len(unique_labels)))
        for i, label in enumerate(labels):
            y_onehot[i, label_to_index[label]] = 1
        
        return y_onehot

class Plotter():
    def __init__(self,plt,sns):
        self.plt = plt
        self.sns = sns

    def bar_plot(self,data,x_feature_name, y_feature_name, fig_dimx=10,fig_dimy=8, title="Count Plot", x_label=" X axis", y_label = "Y axis"):
        self.plt.figure(figsize=(fig_dimx,fig_dimy))
        ax = self.sns.barplot(
            x=x_feature_name,
            y=y_feature_name,
            data=data,
            palette='pastel',
            legend=False
        )

        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        self.plt.xticks(rotation = 90)

        for p in self.plt.gca().patches:
            self.plt.gca().annotate(f'{p.get_height():.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                ha='center', fontsize=10)

        self.plt.show()

    def plot_dict(self, dictionary, type="bar", fig_dimx = 10, fig_dimy = 10, title="Count", x_label=" X axis", y_label = "Y axis"):
        keys = list(dictionary.keys())
        values = list(dictionary.values())

        # Generate random colors for each bar
        colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(len(keys))]

        self.plt.figure(figsize=(fig_dimx,fig_dimy))

        if type == "bar":
            self.plt.bar(keys, values, color=colors)

            for i, value in enumerate(values):
                self.plt.text(keys[i], value + value * 0.02, str(value), ha='center', fontsize=10)
            
            plt.xticks(keys)
        elif type == "pie":
            plt.pie(values, labels=keys, colors=colors,autopct='%1.1f%%')
        else:
            print("Invalid Type of plot!")
            return

        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        plt.tight_layout()
        plt.legend()
        self.plt.show()

    def show_plot_for_corr_pairs(self, corr_pairs, fig_dimx = 10, fig_dimy = 10, title="Count", x_label=" X axis", y_label = "Y axis"):
        processor = Processor()
        corr_dict = processor.get_correlatd_count(corr_pairs=corr_pairs)

        self.plot_dict(corr_dict, fig_dimx=fig_dimx, fig_dimy=fig_dimy,title=title,x_label=x_label,y_label=y_label)

    def plot_Train_validation_curves(self, history):
        plt.figure(figsize=(10,5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs. Validation Loss Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs. Validation Accuracy Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self,cf_matrix, labels):
        unique_labels = sorted(labels.unique())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def plot_grouped_bar_chart(self, datasets, models, mlp_metrics, lstm_metrics,title, y_label, left_margin = 0.001, right_margin = 0.001):
        x = np.arange(len(datasets))
        width = 0.20

        fig, ax = plt.subplots()

        bar1 = ax.bar(x - width/2, mlp_metrics, width, label = "MLP")
        bar2 = ax.bar(x + width/2, lstm_metrics, width, label = "LSTM")

        for bar in bar1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2 - left_margin, height),
                        xytext=(0, 2),  
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bar2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2 + right_margin, height),
                        xytext=(0, 2),  
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_ylabel(f"{y_label} --->")
        ax.set_xlabel("Datasets --->")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        # ax.legend()
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.show()


    def plot_class_wise_barchart(self, datasets, class_f1_scores,title = "",y_label = "" , model_name = ""):
        """
        Parameters:
        - datasets: List of dataset names (e.g., ['Full', 'Reduced'])
        - class_f1_scores: List of lists containing F1 scores for each class
        Format: [[dataset1_class0, dataset1_class1, dataset1_class3], ...]
        - classes: List of original class labels (default [0, 1, 3])
        """

        classes = [0,1,3]
        title= f"{title}: {model_name}"
        y_label=y_label
        figsize=(10, 6)
        bar_width=0.25

        x = np.arange(len(datasets))
        fig, ax = plt.subplots(figsize=figsize)

        # Create bars for each class
        for i, cls in enumerate(classes):
            offsets = bar_width * (i - len(classes)/2 + 0.5)
            scores = [scores[i] for scores in class_f1_scores]
            bars = ax.bar(x + offsets, scores, bar_width, label=f'Class {cls}')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar_width/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_ylabel(f"{y_label} --->")
        ax.set_xlabel("Datasets --->")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


class Metric():
    def __init__(self, fold_accuracies, all_y_true, all_y_pred, predicted_class_probabilities, time_taken):
        self.mean_accuracy = np.mean(fold_accuracies)
        self.std_deviation = np.std(fold_accuracies)
        self.class_metrics = precision_recall_fscore_support(all_y_true, all_y_pred, average=None)
        self.macro_f1 = f1_score(all_y_true, all_y_pred, average='macro')
        self.cf_matrix = confusion_matrix(all_y_true, all_y_pred)
        self.precision = self.class_metrics[0]
        self.recall = self.class_metrics[1]
        self.f1_score = self.class_metrics[2]
        self.time_taken = time_taken

        self.fold_accuracies = fold_accuracies
        self.all_y_true = all_y_true
        self.all_y_pred = all_y_pred
        self.predicted_class_probabilities = predicted_class_probabilities

    def print_metrics(self):
        print("\nOverall Metrics: \n")
        print(f"--Mean Accuracy: {self.mean_accuracy * 100}")
        print(f"--Standard Deviation: {self.std_deviation * 100}")
        print(f"--Macro F1-Score: {self.macro_f1:.4f}")
        print()

        print("Class-wise Metrics:")
        print(f"--Precision: {self.precision}")
        print(f"--Recall: {np.array2string(self.recall, precision=6)}")
        print(f"--F1-Score: {self.f1_score}")

        print(f"\n\nTime Taken By the model: {self.time_taken} seconds \n\n")

    def get_confusion_matrix(self):
        return self.cf_matrix
    
    def get_model_time(self):
        return self.time_taken

class Models():
    def __init__(self):
        self.weights_folder = "weights"
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

    def RandomForest(self, data_without_labels, labels, number_of_trees):
        start_time = time.time()

        X = data_without_labels
        y = labels

        print(f"Number of Input Features: {X.shape[1]}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        unique_labels = sorted(y.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        y_mapped = np.array([label_mapping[label] for label in y])

        y_one_hot = to_categorical(y_mapped, num_classes=len(unique_labels))

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        print(f"Total number of samples: {len(X)}")

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_mapped[train_index], y_mapped[test_index]  # classifier expects 1D labels

            rf_classifier = RandomForestClassifier(n_estimators=number_of_trees, random_state=42)
            rf_classifier.fit(X_train, y_train)

            y_pred = rf_classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

            print(f"Fold {fold+1}:")
            print(f'Accuracy: {accuracy}')
            # print(classification_report(y_test, y_pred, target_names=unique_labels))
            print()

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_probs.append(y_pred)

        end_time = time.time()
        total_time = end_time - start_time

        # Generate overall metrics
        metrics = Metric(fold_accuracies=fold_accuracies, all_y_true=all_y_true, all_y_pred=all_y_pred, predicted_class_probabilities=all_y_probs, time_taken=total_time)

        # Plot confusion matrix
        cf_matrix = metrics.get_confusion_matrix()
        Plotter(None, None).plot_confusion_matrix(cf_matrix=cf_matrix, labels=labels)

        return metrics
    
    def MLP(self, data_without_labels, labels, epochs, dataset_name):
        start_time = time.time()

        X = data_without_labels
        y = labels

        print(f"Number of Input Features: {data_without_labels.shape[1]}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        unique_labels = sorted(y.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        label_inverse_mapping = {v: k for k, v in label_mapping.items()} 

        y_mapped = np.array([label_mapping[label] for label in y])

        y_one_hot = to_categorical(y_mapped, num_classes=len(unique_labels))

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        best_accuracy = 0 

        print(f"Total number of samples: {len(data_without_labels)}")
        for fold, (train_index, test_index) in enumerate(kf.split(data_without_labels)):
            X_train, X_test = data_without_labels[train_index], data_without_labels[test_index]
            y_train,y_test = y_one_hot[train_index], y_one_hot[test_index]

            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(len(unique_labels), activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)
            loss, accuracy = model.evaluate(X_test, y_test)
            fold_accuracies.append(accuracy)

            print(f"Fold {fold+1}:")
            print(f'Accuracy: {accuracy}%')

            y_pred = model.predict(X_test)
            y_pred_original = np.vectorize(label_inverse_mapping.get)(np.argmax(y_pred, axis=1))
            
            y_test_original = y[test_index]
            
            all_y_true.extend(y_test_original)
            all_y_pred.extend(y_pred_original)
            all_y_probs.extend(y_pred)

            # y_pred_mapped = np.argmax(y_pred, axis=1)

            # y_test_mapped = np.argmax(y_test, axis=1)
            
            # all_y_true.extend(y_test_mapped)
            # all_y_pred.extend(y_pred_mapped)

            Plotter(None, None).plot_Train_validation_curves(history=history)

            # Save the weights of the model if accuracy is more

            if accuracy > best_accuracy:
                model.save_weights(os.path.join(self.weights_folder, f"{dataset_name}_MLP_fold{fold + 1}.weights.h5"))
                best_accuracy = accuracy

            print()

        end_time = time.time()
        time_taken = end_time - start_time
        # Make the metrics
        metrics = Metric(fold_accuracies=fold_accuracies, all_y_true=all_y_true, all_y_pred=all_y_pred, predicted_class_probabilities=all_y_probs, time_taken=time_taken)

        # Get the confusion matrix and plot it
        cf_matrix = metrics.get_confusion_matrix()
        Plotter(None,None).plot_confusion_matrix(cf_matrix=cf_matrix, labels=labels)

        tf.keras.backend.clear_session()

        return metrics
        

    def LSTM(self, data_without_labels, labels, epochs, timesteps, dataset_name):
        start_time = time.time()

        unique_labels = sorted(labels.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        label_inverse_mapping = {v: k for k, v in label_mapping.items()}

        y_mapped = np.array([label_mapping[label] for label in labels])

        y_one_hot = to_categorical(y_mapped, num_classes=len(unique_labels))

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # tscv = TimeSeriesSplit(n_splits=5)
        kf = KFold(n_splits=5, shuffle=False)

        # number_of_classes = y_one_hot.shape[1]

        total_features = data_without_labels.shape[1]
        assert total_features % timesteps == 0, "Number of columns must be divisible by timesteps"

        number_of_features = total_features // timesteps
        X_reshaped = data_without_labels.reshape(-1, timesteps, number_of_features)

        best_accuracy = 0

        for fold, (train_index, test_index) in enumerate(kf.split(X_reshaped)):
            X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
            y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

            # y_train_mapped = np.vectorize(label_mapping.get)(y_train)
            # y_test_mapped = np.vectorize(label_mapping.get)(y_test)
            # y_train_onehot = to_categorical(y_train_mapped, num_classes=len(unique_labels))
            # y_test_onehot = to_categorical(y_test_mapped, num_classes=len(unique_labels))

            lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(timesteps, number_of_features)),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(32),
                BatchNormalization(),
                Dropout(0.2),

                Dense(len(unique_labels), activation='softmax')
            ])
            

            lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size = 32, validation_split=0.1, shuffle=False, verbose=0)
            loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)

            fold_accuracies.append(accuracy)

            print(f"-------- Fold : {fold + 1}")

            print(f"Accuracy: {accuracy}")

            y_pred_probs = lstm_model.predict(X_test)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_pred_original = np.vectorize(label_inverse_mapping.get)(y_pred_classes)

            y_test_classes = np.argmax(y_test, axis=1)
            y_test_original = np.vectorize(label_inverse_mapping.get)(y_test_classes)


            all_y_true.extend(y_test_original)
            all_y_pred.extend(y_pred_original)
            all_y_probs.extend(y_pred_probs)

            # Save the weights of the model if accuracy is more
            if accuracy > best_accuracy:
                lstm_model.save_weights(os.path.join(self.weights_folder, f"{dataset_name}_LSTM_fold{fold + 1}.weights.h5"))
                best_accuracy = accuracy

            Plotter(None, None).plot_Train_validation_curves(history=history)

            print()


        end_time = time.time()
        time_taken = end_time - start_time

        # Make the metrics
        metrics = Metric(fold_accuracies=fold_accuracies, all_y_true=all_y_true, all_y_pred=all_y_pred, predicted_class_probabilities=all_y_probs, time_taken=time_taken)

        # Get the confusion matrix and plot it
        cf_matrix = metrics.get_confusion_matrix()
        Plotter(None,None).plot_confusion_matrix(cf_matrix=cf_matrix, labels=labels)

        tf.keras.backend.clear_session()

        return metrics

