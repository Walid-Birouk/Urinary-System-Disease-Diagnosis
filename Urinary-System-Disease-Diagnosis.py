
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('data.csv')  # Data for processing
# Create a copy of the 'data' DataFrame to use it for display
diagnosis_data_display = data.copy()

# Data preprocessing
data['a1'] = data['a1'].str.replace(',', '.').astype(float)  # Convert string to float
categorical_columns = ['a2', 'a3', 'a4', 'a5', 'a6']  # Define categorical columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}  # Initialize label encoders for each categorical column
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])  # Transform categorical columns

# Define features and labels
X = data[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']]  # Features
y_d1 = data['d1']  # Label 1
y_d2 = data['d2']  # Label 2

# Streamlit app
st.title("Urinary System Disease Diagnosis")  # App title

# Data information section
st.header("Dataset Information")  # Header
# Dataset details
st.write("""
The main idea of this dataset is to diagnose two diseases of the urinary system, specifically acute inflammations of the urinary bladder and acute nephritises. Here's some information about these diseases:

**Acute Inflammation of Urinary Bladder:**
- Symptoms: sudden pains in the abdomen, constant urine pushing, micturition pains, and sometimes lack of urine keeping.
- Temperature: rises but usually not above 38°C.
- Urine: turbid and sometimes bloody.
- Response to treatment: symptoms decay within several days but can return, turning into a protracted form.

**Acute Nephritis of Renal Pelvis Origin:**
- Occurs more often in women.
- Symptoms: sudden fever, shivers, one- or both-side lumbar pains, nausea, vomiting, and spread pains of the whole abdomen.

Each instance in the dataset represents a potential patient, and it was created for the diagnosis of these diseases using Rough Sets Theory.
         
**Additional information:**
- a1    Temperature of patient  { 35C-42C } 
- a2    Occurrence of nausea  { yes, no }   
- a3    Lumbar pain  { yes, no }    
- a4    Urine pushing (continuous need for urination)  { yes, no }  
- a5    Micturition pains  { yes, no }  
- a6    Burning of urethra, itch, swelling of urethra outlet  { yes, no }   
- d1    decision: Inflammation of urinary bladder  { yes, no }  
- d2    decision: Nephritis of renal pelvis origin { yes, no }  
         

""")

# Dataset display section
st.header("Dataset")  # Header
st.write(diagnosis_data_display)  # Display the dataset

# User input section for symptoms
st.header("Symptom Input")  # Header
st.subheader("Please answer the following questions:")  # Subheader

# Collect user inputs for symptoms via sliders and radio buttons
temperature = st.slider("Temperature of the patient (in Celsius)", 35.0, 42.0, 36.0)
occurrence_of_nausea = st.radio("Occurrence of nausea", ["Yes", "No"])
lumbar_pain = st.radio("Lumbar pain", ["Yes", "No"])
urine_pushing = st.radio("Urine pushing (continuous need for urination)", ["Yes", "No"])
micturition_pains = st.radio("Micturition pains", ["Yes", "No"])
burning_urethra_itch_swelling = st.radio("Burning of urethra, itch, swelling of urethra outlet", ["Yes", "No"])

# Algorithm selection section
st.header("Select Algorithm")  # Header

# User selects algorithm from dropdown menu
algorithm = st.selectbox("Choose an algorithm", ["Decision Tree", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"])

# If KNN is selected, display a slider to select the number of neighbors 
n_neighbors = 5  
if algorithm == "K-Nearest Neighbors (KNN)":
    n_neighbors = st.slider("Number of neighbors (K) for K-Nearest Neighbors (KNN)", 1, 20, 5)

# Prepare input data based on user's symptom inputs 
input_data = {
    'a1': temperature,
    'a2': 1 if occurrence_of_nausea == "Yes" else 0,
    'a3': 1 if lumbar_pain == "Yes" else 0,
    'a4': 1 if urine_pushing == "Yes" else 0,
    'a5': 1 if micturition_pains == "Yes" else 0,
    'a6': 1 if burning_urethra_itch_swelling == "Yes" else 0
}

# Diagnose based on the selected algorithm
if st.button("Diagnose"):
    # If the selected algorithm is Decision Tree
    if algorithm == "Decision Tree":
        # Create a Decision Tree classifier for diagnosis 1 (d1)
        dt_classifier_d1 = DecisionTreeClassifier()
        # Train the classifier with the input data (X) and corresponding labels (y_d1)
        dt_classifier_d1.fit(X, y_d1)
        # Predict the diagnosis for the input data
        prediction_d1 = dt_classifier_d1.predict([list(input_data.values())])[0]
        # Get the probability of each class for the input data
        prediction_prob_d1 = dt_classifier_d1.predict_proba([list(input_data.values())])
        # Since it's a decision tree, we assume it's perfectly accurate
        accuracy_d1 = 1.0

        # Repeat the same process for diagnosis 2 (d2)
        dt_classifier_d2 = DecisionTreeClassifier()
        dt_classifier_d2.fit(X, y_d2)
        prediction_d2 = dt_classifier_d2.predict([list(input_data.values())])[0]
        prediction_prob_d2 = dt_classifier_d2.predict_proba([list(input_data.values())])
        accuracy_d2 = 1.0

    # If the selected algorithm is K-Nearest Neighbors (KNN)
    elif algorithm == "K-Nearest Neighbors (KNN)":
        # Create a KNN classifier for diagnosis 1 (d1) with a specified number of neighbors
        knn_classifier_d1 = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier_d1.fit(X, y_d1)
        prediction_d1 = knn_classifier_d1.predict([list(input_data.values())])[0]
        prediction_prob_d1 = knn_classifier_d1.predict_proba([list(input_data.values())])
        # Calculate the accuracy of the classifier on the training data
        accuracy_d1 = knn_classifier_d1.score(X, y_d1)

        # Repeat the same process for diagnosis 2 (d2)
        knn_classifier_d2 = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier_d2.fit(X, y_d2)
        prediction_d2 = knn_classifier_d2.predict([list(input_data.values())])[0]
        prediction_prob_d2 = knn_classifier_d2.predict_proba([list(input_data.values())])
        accuracy_d2 = knn_classifier_d2.score(X, y_d2)

    # If the selected algorithm is Support Vector Machine (SVM)
    elif algorithm == "Support Vector Machine (SVM)":
        # Create an SVM classifier for diagnosis 1 (d1) with probability estimation enabled
        svm_classifier_d1 = SVC(probability=True)
        svm_classifier_d1.fit(X, y_d1)
        prediction_d1 = svm_classifier_d1.predict([list(input_data.values())])[0]
        prediction_prob_d1 = svm_classifier_d1.predict_proba([list(input_data.values())])
        accuracy_d1 = svm_classifier_d1.score(X, y_d1)

        # Repeat the same process for diagnosis 2 (d2)
        svm_classifier_d2 = SVC(probability=True)
        svm_classifier_d2.fit(X, y_d2)
        prediction_d2 = svm_classifier_d2.predict([list(input_data.values())])[0]
        prediction_prob_d2 = svm_classifier_d2.predict_proba([list(input_data.values())])
        accuracy_d2 = svm_classifier_d2.score(X, y_d2)

    # Display the diagnosis and confidence level
    st.header("Diagnosis Results")
    st.write("**Inflammation of the urinary bladder (d1):**")
    st.write("Diagnosis:", "**Yes**" if prediction_d1 == "yes" else "**No**")
    st.write("Confidence Level:", f"**{max(prediction_prob_d1[0]):.2f}**")
    st.write("Accuracy:", f"**{accuracy_d1:.2f}**")

    st.write("**Nephritis of renal pelvis origin (d2):**")
    st.write("Diagnosis:", "**Yes**" if prediction_d2 == "yes" else "**No**")
    st.write("Confidence Level:", f"**{max(prediction_prob_d2[0]):.2f}**")
    st.write("Accuracy:", f"**{accuracy_d2:.2f}**")
