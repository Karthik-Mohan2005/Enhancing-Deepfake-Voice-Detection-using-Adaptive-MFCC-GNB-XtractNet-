## Implementation

## Project Execution Guide

This section explains the complete procedure for downloading, installing, configuring, and executing the Deepfake Voice Detection System from the GitHub repository.

The system is implemented using Python and Flask. The trained model is integrated with the Flask application to classify uploaded speech audio as REAL or FAKE.

---

## 1. Clone the GitHub Repository

git clone https://github.com/Karthik-Mohan2005/Enhancing-Deepfake-Voice-Detection-using-Adaptive-MFCC-GNB-XtractNet-.git
    After the repository has been successfully cloned, navigate to the project directory:


cd Enhancing-Deepfake-Voice-Detection-using-Adaptive-MFCC-GNB-XtractNet-

## 2. Navigate to the Backend Directory
The main Python application and dependency files are located inside the backend directory.
Run:

cd backend

The backend directory contains the main application files, trained models, dataset, templates, static files, and other required resources.

## 3. Create a Virtual Environment
Create a Python virtual environment to install the project dependencies separately from the system Python environment.
Run:

python -m venv venv

This creates a virtual environment named venv inside the backend directory.

## 4. Activate the Virtual Environment
Windows PowerShell
Run:

venv\Scripts\activate

Windows Command Prompt
Run:

venv\Scripts\activate.bat

After successful activation, the terminal will display (venv) before the current directory.
Example:
(venv) PS D:\Enhancing-Deepfake-Voice-Detection-using-Adaptive-MFCC-GNB-XtractNet-\backend>

## 5. Install Required Dependencies
The project contains a requirements.txt file inside the backend directory.

Run:
 
 cd backend

Install all required Python packages using:


pip install -r requirements.txt

The required libraries include the packages necessary for:
- Audio processing
- MFCC feature extraction
- Machine learning
- Model loading
- Flask web application
- Numerical and data processing
Wait until the installation is completed successfully.


## 6. Run the Flask Application
After installing all the required dependencies, execute the live Flask application.
Run:

python app_live.py

The Flask server will start and the terminal will display a local server address.
The address will normally be:

http://127.0.0.1:5000

The terminal may display:
* Running on http://127.0.0.1:5000

## 7. Open the Web Application
Open any web browser such as Google Chrome, Microsoft Edge, or Mozilla Firefox.
Enter the following address in the browser:

http://127.0.0.1:5000

The Deepfake Voice Detection System web interface will open.

## 8. Upload an Audio File
After opening the web application:
1. Select the audio upload option.
2. Choose an audio file from the local computer.
3. Upload the audio file.
4. The system receives the uploaded audio.
5. The audio is processed by the backend.
6. MFCC features are extracted from the audio.
7. The trained machine-learning model analyzes the extracted features.
8. The system generates the final prediction.
9. The result is displayed on the web interface.
