import os
import csv
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, precision_recall_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Activation,
                                     LSTM, Input, concatenate, GlobalAveragePooling1D,
                                     LayerNormalization, MultiHeadAttention, Add, Flatten,
                                     Concatenate)
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers




# Define a custom callback for timing
class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.logs=[]
    def on_train_begin(self, logs={}):
        self.start_time = time.time()
    def on_train_end(self, logs={}):
        self.logs.append(time.time() - self.start_time)



label_encoder = LabelEncoder()

# Function to normalize data
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

# Function to load features
def load_features(folder, labels):
    X = []
    y = []

    for label in labels:
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith('.npy'):
                    feature_path = os.path.join(label_folder, filename)
                    feature = np.load(feature_path)
                    X.append(feature)
                    y.append(label)

    return np.array(X), np.array(y)

# Define directory variables
directory_work = os.getcwd()
directory_feature = os.path.join(directory_work, 'ADandNAD_Feature')
model_name = 'scenario_2_Tranformer_1'
result_out = os.path.join(directory_work, model_name)
os.makedirs(result_out, exist_ok=True)

# Learning Rate Reduction Callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.000001, cooldown=2)

# Adam Optimizer
# optimizer = Adam(learning_rate=0.0001)

# Define the categories
categories = ['AD','NAD']
num_categories = len(categories)
num_classes=num_categories


train_resnet152_folder = os.path.join(directory_feature, 'train_Deep_Hand_Crafted_Features', 'resnet152_features')
test_resnet152_folder = os.path.join(directory_feature, 'test_Deep_Hand_Crafted_Features', 'resnet152_features')

train_resnet101_folder = os.path.join(directory_feature, 'train_Deep_Hand_Crafted_Features', 'resnet101_features')
test_resnet101_folder = os.path.join(directory_feature, 'test_Deep_Hand_Crafted_Features', 'resnet101_features')

train_resnet50_folder = os.path.join(directory_feature, 'train_Deep_Hand_Crafted_Features', 'vgg19_features')
test_resnet50_folder = os.path.join(directory_feature, 'test_Deep_Hand_Crafted_Features', 'vgg19_features')


# 5. resnet152_features
train_resnet152_features, train_resnet152_labels = load_features(train_resnet152_folder, categories)
test_resnet152_features, test_resnet152_labels = load_features(test_resnet152_folder, categories)

# 6. resnet101_features
train_resnet101_features, train_resnet101_labels = load_features(train_resnet101_folder, categories)
test_resnet101_features, test_resnet101_labels = load_features(test_resnet101_folder, categories)

# 7. resnet50_features
train_resnet50_features, train_resnet50_labels = load_features(train_resnet50_folder, categories)
test_resnet50_features, test_resnet50_labels = load_features(test_resnet50_folder, categories)



# Now process the test labels
test_labels_encoded = label_encoder.fit_transform(test_resnet152_labels)
test_labels_categorical = to_categorical(test_labels_encoded, num_classes=num_categories)



# Initialize a scaler for each feature type
scaler = StandardScaler()
scaler_resnet152 = StandardScaler()
scaler_resnet101 = StandardScaler()
scaler_resnet50 = StandardScaler()

resnet152_features_dim = train_resnet152_features.shape[1]
resnet101_features_dim = train_resnet101_features.shape[1]
resnet50_features_dim = train_resnet50_features.shape[1]


train_resnet152_features = scaler_resnet152.fit_transform(train_resnet152_features.reshape(-1, resnet152_features_dim)).reshape(-1, 1, resnet152_features_dim)
test_resnet152_features = scaler_resnet152.transform(test_resnet152_features.reshape(-1, resnet152_features_dim)).reshape(-1, 1, resnet152_features_dim)

train_resnet101_features = scaler_resnet152.fit_transform(train_resnet101_features.reshape(-1, resnet101_features_dim)).reshape(-1, 1, resnet101_features_dim)
test_resnet101_features = scaler_resnet152.transform(test_resnet101_features.reshape(-1, resnet101_features_dim)).reshape(-1, 1, resnet101_features_dim)

train_resnet50_features = scaler_resnet152.fit_transform(train_resnet50_features.reshape(-1, resnet50_features_dim)).reshape(-1, 1, resnet50_features_dim)
test_resnet50_features = scaler_resnet152.transform(test_resnet50_features.reshape(-1, resnet50_features_dim)).reshape(-1, 1, resnet50_features_dim)


# Concatenate the feature sets for SMOTE
train_features_combined = np.concatenate([
    train_resnet152_features.reshape(-1, resnet152_features_dim),
    train_resnet101_features.reshape(-1, resnet101_features_dim),
    train_resnet50_features.reshape(-1, resnet50_features_dim)
    ], axis=1)


train_labels_encoded = label_encoder.fit_transform(train_resnet152_labels)  


# Ensure the labels are consistent across splits
X_test_combined = np.concatenate([test_resnet152_features, test_resnet101_features, test_resnet50_features], axis=-1)
y_test_combined = test_labels_categorical 

# Splitting the combined test set into validation and new test sets (50% validation, 50% test)
X_test_combined, X_val_combined, y_test_combined, y_val_combined = train_test_split(
    X_test_combined, y_test_combined, test_size=0.5, random_state=42)

# Now, separate the validation set back into individual feature sets
val_resnet152_features = X_val_combined[:, :, :resnet152_features_dim]
val_resnet101_features = X_val_combined[:, :, resnet152_features_dim:resnet152_features_dim+resnet101_features_dim]
val_resnet50_features = X_val_combined[:, :, -resnet50_features_dim:]

test_resnet152_features = X_test_combined[:, :, :resnet152_features_dim]
test_resnet101_features = X_test_combined[:, :, resnet152_features_dim:resnet152_features_dim+resnet101_features_dim]
test_resnet50_features = X_test_combined[:, :, -resnet50_features_dim:]


# Apply SMOTE
smote = SMOTE()
train_features_combined_smote, train_labels_smote = smote.fit_resample(train_features_combined, train_labels_encoded)

# Number of samples after SMOTE
num_samples_after_smote = train_features_combined_smote.shape[0]

cumulative_dim_resnet152 = resnet152_features_dim
cumulative_dim_resnet101 = cumulative_dim_resnet152 + resnet101_features_dim
cumulative_dim_resnet50 = cumulative_dim_resnet101 + resnet50_features_dim


train_resnet152_features = train_features_combined_smote[:, :cumulative_dim_resnet152].reshape(-1, 1, resnet152_features_dim)
train_resnet101_features = train_features_combined_smote[:, cumulative_dim_resnet152:cumulative_dim_resnet101].reshape(-1, 1, resnet101_features_dim)
train_resnet50_features = train_features_combined_smote[:, cumulative_dim_resnet101:].reshape(-1, 1, resnet50_features_dim)

# Make sure the labels are also reshaped accordingly
train_labels_categorical = to_categorical(train_labels_smote, num_classes=num_categories)

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):

    # Layer normalization 1
    x = LayerNormalization(epsilon=1e-6)(inputs)
    
    # Multi-head self-attention
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    
    # Skip connection
    x = Add()([x, attn_output])
    
    # Layer normalization 2
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    
    # Skip connection
    x = Add()([x, ff_output])
    return x


def build_transformer_model(hp):
    inputs = []
    for feature_dim in [resnet152_features_dim, resnet101_features_dim, resnet50_features_dim]:
        inputs.append(Input(shape=(1, feature_dim)))
    
    # Transformer blocks
    encoded_features = []
    for input_layer in inputs:
        x = input_layer
        for _ in range(hp.Int('num_transformer_blocks', 1, 4, step=1)):
            x = transformer_encoder_block(
                x,
                head_size=hp.Int('head_size', 32, 128, step=32),
                num_heads=hp.Int('num_heads', 2, 8, step=2),
                ff_dim=hp.Int('ff_dim', 32, 128, step=32),
                dropout=hp.Float('dropout', 0, 0.5, step=0.1)
            )
        encoded_features.append(x)
    
    # Concatenate all encoded features
    x = layers.Concatenate()(encoded_features) if len(encoded_features) > 1 else encoded_features[0]
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Flatten()(x)
    x = Dropout(hp.Float('final_dropout', 0, 0.5, step=0.1))(x)
    
    # Additional dense layers
    for units in [hp.Int('units', min_value=32, max_value=512, step=32)]:
        x = Dense(units, activation='relu')(x)
        x = Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    
    # Output layer
    outputs = Dense(num_categories, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



# Instantiate the tuner
tuner = kt.Hyperband(
    build_transformer_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_tuner_dir',
    project_name = model_name
)


# Combine all feature sets into a list
all_train_features = [
            train_resnet152_features,
            train_resnet101_features,
            train_resnet50_features
]

all_test_features = [
            test_resnet152_features,
            test_resnet101_features,
            test_resnet50_features
            ]


# Define the batch sizes and epochs you want to try
batch_size = 64
epoch = 100

# Store the results
training_results = []
classification_reports = []
model_summary = []

tuner.search(
            all_train_features, 
            train_labels_categorical,
            validation_data = ([val_resnet152_features, val_resnet101_features, val_resnet50_features], y_val_combined),
            epochs=epoch,
            batch_size=batch_size
        )
        # Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
with open(os.path.join(result_out, f'{model_name}_hyperparameters.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hyperparameter', 'Value'])
    for hparam in best_hps.values:
        writer.writerow([hparam, best_hps.get(hparam)])
        
# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

plot_model(model, to_file=os.path.join(result_out, f'{model_name}_bs_{batch_size}_ep_{epoch}.png'), show_shapes=True)

# Summarize the model

model.summary(print_fn=lambda x: model_summary.append(x))

# Define the path to the CSV file where you want to save the model summary
model_summary_csv_file = os.path.join(result_out, f'{model_name}_model_summary_bs_{batch_size}_ep_{epoch}.csv')

# Save the model summary to a CSV file
with open(model_summary_csv_file, 'w') as f:
    f.write("\n".join(model_summary))


test_metrics_data = []


# Callbacks List without ModelCheckpoint
timing_callback = TimingCallback()
callbacks = [learning_rate_reduction, timing_callback]

# Start training the model
history = model.fit(
    all_train_features, 
    train_labels_categorical,
    validation_data=([val_resnet152_features, val_resnet101_features, val_resnet50_features], y_val_combined),
    epochs = epoch, 
    batch_size = batch_size, 
    callbacks = callbacks,
    verbose=1
)

# Calculate total training time
total_training_time = sum(timing_callback.logs)
print(f"Total training time: {total_training_time:.2f} seconds")

# y_pred model predictions
y_pred = model.predict([test_resnet152_features, test_resnet101_features, test_resnet50_features])

# Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
# Print shapes for debugging
print("Prediction shape:", y_pred_classes.shape)

# Convert categorical labels back to class labels for comparison
y_true_classes = np.argmax(y_test_combined, axis=1)  # Correct this line if y_test_combined is not the correct variable for true labels
print("True labels shape:", y_true_classes.shape)


# Model evaluation with the correctly split test sets
test_loss, test_accuracy = model.evaluate(
    [test_resnet152_features, test_resnet101_features, test_resnet50_features], 
    y_test_combined
)


# Calculating metrics
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
fpr, tpr, _ = roc_curve(y_true_classes, y_pred[:, 1])


# Finding the best epoch and corresponding loss
best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 because epochs are 1-indexed in the logs
best_val_loss = np.min(history.history['val_loss'])

# model.save(os.path.join(result_out, f'{model_name}_model.h5'))
print("Model saved successfully.")


# Assuming `y_pred` and `y_true_classes` are defined as shown previously
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='macro')
recall = recall_score(y_true_classes, y_pred_classes, average='macro')
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

cm = confusion_matrix(y_true_classes, y_pred_classes)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

sensitivity = TP / float(TP + FN)
specificity = TN / float(TN + FP)
auc_score = roc_auc_score(y_true_classes, y_pred[:, 1])

fpr, tpr, thresholds = roc_curve(y_true_classes, y_pred[:, 1])
roc_auc = auc(fpr, tpr)


print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
print(f"AUC Score: {auc_score}")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss}")


test_metrics_data.append({
    'Model Name': model_name,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Sensitivity (Recall)': sensitivity,
    'Specificity': specificity,
    'AUC Score (AD)': auc_score,
    'Loss': test_loss,
    'Best Epoch': best_epoch,
    'Total Training Time (s)': total_training_time
})

# Convert the dictionary into a DataFrame
metrics_df = pd.DataFrame(test_metrics_data)

# Save the DataFrame to a CSV file
metrics_filename = os.path.join(result_out, f'{model_name}_test_metrics_data.csv')
metrics_df.to_csv(metrics_filename, index=False)
print(f"Metrics saved to {metrics_filename}")


# Corrected Accuracy and Loss Plots
fig1, ax1 = plt.subplots(figsize=(8, 4))
epochs_range = range(1, len(history.history['accuracy']) + 1)  # Corrected to dynamically determine the number of epochs

# Plot training and validation accuracy
ax1.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
ax1.legend(loc='lower right')

# Save the accuracy plot
fig1.savefig(os.path.join(result_out, f'{model_name}_accuracy.png'))

# Loss Plot
fig2, ax2 = plt.subplots(figsize=(8, 4))
# Plot training and validation loss
ax2.plot(epochs_range, history.history['loss'], label='Train Loss')
ax2.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
ax2.legend(loc='upper right')
# Save the loss plot
fig2.savefig(os.path.join(result_out, f'{model_name}_loss.png'))
plt.close('all')  # Close all figures to free memory


# Visualization
plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["AD", "NAD"], yticklabels=["AD", "NAD"])
plt.savefig(os.path.join(result_out, f'{model_name}_confusion_matrix.png'))
plt.close()


# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve for 'AD' (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig(os.path.join(result_out, f'{model_name}_ROC_curve_AD.png'))  # Save the ROC curve plot for 'AD'
plt.close()


# Assuming label_encoder is your LabelEncoder instance and it has been fitted
classes = list(label_encoder.classes_)

class_0 = classes[0]  # This will give you the class encoded as 0
class_1 = classes[1]  # This will give you the class encoded as 1

print(f"Class 0: {class_0}")  # This could be 'AD' or 'NAD'
print(f"Class 1: {class_1}")  # This could be 'AD' or 'NAD', the opposite of class 0



