# HumanActivity_Recognition


### LSTM Model Training, Evaluation, and Visualizatio

1. **Data Preparation**:
   - The dataset is first split into training (`X_train`, `Y_train`) and testing sets (`X_test`, `Y_test`) using a train-test split with 80% for training and 20% for testing.
   - Input features are reshaped to fit the LSTM model structure. This reshaping ensures that the data is fed correctly into the network, maintaining the proper dimensions required by LSTM cells.

2. **LSTM Model Definition**:
   - The LSTM model is defined with two hidden layers, each using a BasicLSTMCell in TensorFlow.
   - The model uses ReLU activation for the hidden layer and connects it to LSTM cells that process sequential data over time steps.
   - The weight matrices (`W`) and biases for both the hidden and output layers are initialized with random values. The output layer is responsible for predicting class labels based on the learned sequence patterns.

3. **Training Process**:
   - The training loop runs for 50 epochs, where in each epoch, the model processes the data in batches (size of 1024) to optimize performance.
   - For each epoch, the model's optimizer adjusts weights and biases based on the gradient of the loss function, improving the model’s performance.
   - Training and testing accuracies and losses are calculated and stored in a `history` dictionary for further analysis. The training accuracy measures how well the model is learning, while test accuracy measures how well it generalizes to unseen data.

4. **Model Evaluation**:
   - After completing 50 epochs, the final accuracy and loss on the test data are displayed. These metrics provide a quick overview of the model’s performance.
   - The `history` dictionary, which tracks the training and testing accuracies and losses across all epochs, is used for visualization. This helps in identifying whether the model is underfitting or overfitting.

5. **Visualization**:
   - A line graph is plotted to show the progression of training and testing losses and accuracies over the 50 epochs. This provides an intuitive way to visualize the model's learning process and see improvements in performance over time.
   - A confusion matrix is generated to provide a detailed breakdown of the model’s classification performance, comparing predicted labels with true labels. This is visualized as a heatmap using Seaborn, giving a clear picture of how well the model is performing across different classes.

6. **Confusion Matrix**:
   - The confusion matrix helps evaluate classification accuracy by showing how many instances of each class were correctly or incorrectly predicted.
   - True labels are compared with the predicted ones, allowing you to identify any misclassifications and evaluate how well the model performs on each class.

