# Urinary System Disease Diagnosis Streamlit App

This README provides an overview of the Streamlit application designed for diagnosing urinary system diseases. The app utilizes a dataset for diagnosing two specific diseases: acute inflammations of the urinary bladder and acute nephritises. Through a user-friendly interface, users can input symptoms, choose a diagnostic algorithm, and receive a diagnosis along with the confidence level of the prediction.

## Application Workflow

### Data Loading and Preprocessing

The app starts by loading a dataset from `data.csv`, which contains records of patient symptoms and diagnoses for the two diseases. The data undergoes preprocessing to convert string values to floats and encode categorical variables using Label Encoding. This step ensures that the data is in the right format for machine learning models.

### User Interface

The app's user interface is divided into several sections:

- **Dataset Information**: Provides an overview of the diseases, symptoms, and the structure of the dataset. This section educates users about the conditions the app can diagnose and the type of data it uses.
  
- **Dataset Display**: Shows the actual dataset used for diagnosis, allowing users to understand the data's format and contents.
  
- **Symptom Input**: Users input their symptoms or the patient's symptoms using sliders and radio buttons. This section is crucial for collecting user data for diagnosis.

- **Algorithm Selection**: Users can select one of three algorithms for diagnosis: Decision Tree, K-Nearest Neighbors (KNN), or Support Vector Machine (SVM). For KNN, users also specify the number of neighbors.

### Diagnosis Process

Based on the user's input and the selected algorithm, the app predicts whether the user has an inflammation of the urinary bladder or nephritis of the renal pelvis origin. The diagnosis includes the prediction (Yes/No) and the confidence level of the prediction. For KNN and SVM algorithms, the accuracy of the model on the training data is also displayed.

### Machine Learning Models

The app uses three machine learning models:

- **Decision Tree Classifier**: A model that uses a decision tree to make predictions. It is simple and interpretable but may overfit on complex datasets.
  
- **K-Nearest Neighbors (KNN)**: A model that classifies instances based on the majority vote of the nearest neighbors. The number of neighbors (K) affects its performance.
  
- **Support Vector Machine (SVM)**: A powerful classifier that finds the optimal hyperplane for classifying data points into two categories. It works well for high-dimensional data.

Each model is trained with the preprocessed dataset and then used to predict the diseases based on the symptoms inputted by the user.

## Conclusion

The Urinary System Disease Diagnosis Streamlit app is a practical tool for predicting acute inflammations of the urinary bladder and acute nephritises using machine learning models. By providing a simple interface for symptom input and algorithm selection, it makes the diagnostic process accessible to non-expert users. The app not only showcases the application of machine learning in healthcare but also serves as an educational tool for understanding these urinary system diseases.

## Link:
https://urinary-system-disease-diagnosis.streamlit.app/
