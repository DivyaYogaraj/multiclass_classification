#multiclass_classification

Unzip the File
use internet for all installation

Install Python 3.14, then verify the installation:
    Press Windows + R, type `cmd`, and press Enter.
    In Command Prompt, run:
        python --version

Open Visual Studio Code and install the required Python extensions.

Go to File → New Window → Open Folder, then select the `multiclass_classification` folder.

Open the terminal and run:
    py -m venv venv
    venv\Scripts\Activate
    cd deployment
    pip install -r requirements.txt

Run the application:
    uvicorn app:app --reload (Click the Link)

On the output page, Fill the 10 features and click Prediction view the Output.

## ⚠️ Important Files – Project Files

All important dataset, trained model, deployment, and notebook files
used by this project are included in this GitHub repository.

### Important files included in this repository

- `data/raw_dataset.csv`
- `models/encoder.pkl`
- `models/model.pkl`
- `models/scaler.pkl`
- `notebooks/MultiClass_Classification.ipynb`
- `deployment/app.py`
- `deployment/static/index.html`
- `deployment/requirements.txt`

The dataset file is used for training and testing the machine learning
model.

The trained model, encoder, and scaler files are required by the
application for making predictions.

The notebook contains the complete machine learning workflow including
data preprocessing, model training, and evaluation.

The deployment files are used to run the trained model as an application.

The files are **not unnecessary** and should not be deleted from the
project.

### Files not included in this repository

- `.refact/`
- `__pycache__/`

These files are excluded because they contain local development and
Python cache files and are not required for running the project.

> **Note:** The important dataset, trained models, notebook, and
> deployment files are included in this repository.
