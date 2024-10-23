# Fuel-Efficiency-Prediction
The objective of this project is to predict the fuel efficiency (miles per gallon - MPG) of various car models based on factors such as the number of cylinders, engine displacement, horsepower, weight, and acceleration. The project aims to identify which features most significantly affect fuel consumption and help improve MPG values in cars.

**Explanations of Concepts:**

**Data Preprocessing:**

**Data Cleaning:** Handling missing values and ensuring data consistency.

**Feature Scaling:** Using normalization to bring all numerical features to the same scale, which is crucial for improving model performance.

**One-Hot Encoding:** Categorical features (e.g., origin) are converted into numerical format through one-hot encoding, which allows the model to process non-numerical data.

**Sample Model Building:**

**Neural Network:** A feedforward deep neural network (DNN) is built using TensorFlow/Keras to predict the target variable mpg. The model consists of:

Two dense layers with 64 neurons and ReLU activation and similarly tried with different possibilities.
An output layer with one neuron for predicting the MPG value.

**Normalization Layer:** The first layer normalizes the input features to ensure they are on a similar scale.

**Loss Function:** Mean Absolute Error (MAE) is used to minimize the difference between actual and predicted values.

**Optimizer:** Adam optimizer is used to adjust the model weights during training.

**Model Training:**

The model is trained on the training data with a validation split (20%) to monitor performance and avoid overfitting.
The model is trained for 100 epochs, with the validation loss and training loss being tracked.

**Model Evaluation:**

After training, the model's performance is evaluated on the test dataset using metrics like Mean Absolute Error (MAE) and R-squared (R²) score.
R² Score: A high R² score (close to 1) indicates that the model’s predictions explain the variance in the MPG values well.

**Fine-Tuning:**

Fine-tuning involves adjusting the model architecture (number of layers, neurons, activation functions, etc.) and optimizer hyperparameters to get the best results. In this project, different configurations were tested, and the best model used 3 dense layers with the ReLU activation function and the Adam optimizer.

**Project Deliverables:**

**Exploratory Data Analysis (EDA):** Visualizations and summary statistics of the dataset to understand the relationship between features and MPG.

**Trained Neural Network Model:** The final deep learning model trained to predict MPG.

**Model Evaluation:** Evaluation metrics like MAE and R² Score to demonstrate the model’s accuracy.

**Code Implementation:** A complete code implementation using Python, TensorFlow/Keras, and necessary data visualization libraries.

**Model Saving:** The trained model is saved as a .keras file for future use or deployment.

**Prediction and Error Analysis:** Scatter plots comparing predicted MPG values to true values, and histograms showing prediction errors.

**Future Scope:**

**Model Improvement:** Further fine-tuning of the model by experimenting with different neural network architectures, optimizers, and learning rates.

**Alternative Models:** Trying other machine learning models like Random Forest, XGBoost, or Support Vector Machines (SVM) to compare performance.

**Feature Engineering:** Creating new features or interaction terms that could improve model accuracy.

**Real-Time Prediction:** Deploying the trained model in a web application to provide real-time MPG predictions for new cars.

**Transfer Learning:** Using the knowledge from this project to fine-tune models on similar datasets for related predictions (e.g., CO2 emissions, electric car range).

**Screenshots:**

![download (1)](https://github.com/user-attachments/assets/8f014b5e-ca4f-4057-af17-82dea6b6cef3)

![download (2)](https://github.com/user-attachments/assets/029ecf91-6da5-4cd9-92c1-a3bfeb1ba400)

![download](https://github.com/user-attachments/assets/53f4b59f-7a11-4dbc-984e-4159b47ede61)
