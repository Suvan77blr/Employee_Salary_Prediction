# Employee Salary Prediction

A machine learning-based system that predicts whether an individual earns more than $50K per year based on demographic and work-related features.

## Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- joblib  
- streamlit  

## Technologies Implemented

- Supervised Machine Learning (Classification)  
- Logistic Regression, SVM, Random Forest, Gradient Boosting, MLP Classifier  
- Label Encoding and Feature Scaling  
- Model Evaluation using Accuracy, ROC AUC, Precision, Recall, F1-score  
- Interactive Web Interface using Streamlit  
- Model Persistence using Joblib  

## Steps Implemented

- Load and explore the dataset  
- Clean and preprocess the data (handle missing values, encode categories, scale features)  
- Perform feature engineering (log transforms, derived flags)  
- Train multiple classification models and evaluate performance  
- Select the best-performing model based on evaluation metrics  
- Save the final model and encoders using joblib  
- Build a Streamlit web application for user input and predictions  

## Interface Code (Streamlit App)

The `app.py` file contains the code for the user interface using Streamlit. It allows users to input employee details and returns the predicted salary class.

### Local Setup

1. Clone the repository
```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
```

2. Create a virtual environment (optional but recommended)  
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies  
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app  
```bash
streamlit run app.py
```


5. The app will launch in your browser at `http://localhost:8501`

---

## Notes

- Ensure `salary_model.pkl` and `label_encoders.pkl` are present in the project directory before running the app.  
- You can modify the model or preprocessing logic in the corresponding notebook/script files.