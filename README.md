School project using Random forest model to predict student performance from a kaggle dataset

## Requirements

### Software Requirements

* Windows 10 or Windows 11
* Python 3.10 or newer
* Visual Studio Code or another Python-compatible IDE
* pandas
* NumPy
* scikit-learn
* Git/GitLab for source-code management

The required Python libraries can be installed using the provided `requirements.txt` file.

### Hardware Requirements

* Multi-core processor
* Minimum 8 GB RAM
* At least 10 GB of available storage
* Internet connection for downloading the dataset and Python dependencies

### Dataset Requirements

The application requires the `StudentPerformanceFactors.csv` dataset. The CSV file should be placed in the project's root directory or the location specified by the Python application.

The preprocessed dataset, `StudentPerformanceFactors_preprocessed.csv`, is generated each time the script is ran.

## Instructions to Run

1. **Install Python**
   Install Python 3.10 or newer and verify the installation:

   ```bash
   python --version
   ```

2. **Clone the GitLab repository**

   ```bash
   git clone https://gitlab.com/wgu-gitlab-environment/student-repos/mtran135/d683-advanced-ai-and-ml.git
   cd <project-folder>
   ```

3. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

4. **Activate the virtual environment**

   **Windows:**

   ```bash
   .venv\Scripts\activate
   ```

5. **Install the required libraries**

   ```bash
   pip install -r requirements.txt
   ```

6. **Add the dataset**
   Place `StudentPerformanceFactors.csv` in the project's root directory.

7. **Run the application**

   ```bash
   python main.py
   ```

8. **View the results**
   The application will preprocess the dataset, train the Random Forest Regression model, generate exam-score predictions, and display the model's RMSE and R² performance metrics.
