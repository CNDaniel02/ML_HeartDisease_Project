# Machine Learning-Salary Prediction Project

This project predicts salaries based on various features such as region, experience level, job category, and company size. It includes data preprocessing, visualization, model selection, and web-based interface for prediction. 

---

## Project Files
1. **`ds_salaries.csv`**: Dataset containing salary data and related features.
2. **`ML_Salaries.ipynb`**: Jupyter Notebook with all code and visualizations for easier understanding and exploration.
3. **`ridge_model.pkl`**: Trained Ridge Regression model saved using `joblib`.
4. **`templates/`**: Folder containing HTML files for the web interface:
   - `index.html`: Home page for inputting features.
   - `result.html`: Displays the predicted salary.
5. **`app.py`**: Flask application to serve the web interface and make predictions.
6. **`FinalReport.pdf`**: The final report of the project.
7. **`DEMO.mp4`**: The demonstration video of the project.
8. **`README.md`**: Instructions and descriptions for this project. 

---


## How to Run the Project
### **1. Requirements**
Before running the project, the following Python packages are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `flask`

### **2. Dataset**
Make sure the **`ds_salaries.csv`** file is in the same directory as the code files.

### **3. Running the Code**
Run **`ML_Salaries.ipynb`** with Jupyter Notebook.


### **4. Running the Web Application**
- Start the Flask application:
```bash
  python app.py
```

- Open the provided link (usually http://127.0.0.1:5000) in your browser.
-Use the web interface and input features and get salary predictions.
