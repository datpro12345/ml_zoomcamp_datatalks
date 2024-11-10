# MLZoomCamp Midterm Project

## Building & Deploying a Python ML Model for Stroke Predictions

### (Building Different Classification-based Models and Deploying Best Model in AWS Cloud using Flask & Docker)

DataTalks.Club - [Machine Learning Zoomcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) **Midterm Project** - by Alexey Grigorev

---

## Introduction

Stroke prediction is crucial for timely treatment. Using Kaggle’s Stroke Prediction dataset, this project aims to identify factors influencing strokes and deploys a model that predicts stroke risks.

## Data Preparation

- **Source**: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- **Data Processing**:
    - Loaded data using Pandas and NumPy.
    - Cleaned and formatted the dataset (handled missing values, dropped irrelevant columns).
    - Performed exploratory data analysis (EDA) with Matplotlib and Seaborn.
    - Visualized relationships between features like BMI, age, smoking status, and stroke risk.
    - Encoded categorical variables using one-hot encoding.

*Examples of initial data exploration:*

![image](https://user-images.githubusercontent.com/50409210/139657709-7c6ded3c-9b3a-4c35-86db-c4e1a33a67cb.png)

## Exploratory Data Analysis (EDA)

- **Visual Insights**:
    - People with high-risk factors included smokers, those with high BMI or glucose levels, and those who were self-employed.
    - Correlation between numerical/categorical features and stroke was visualized using heatmaps and statistical tests.

*Examples of EDA visualizations:*

![image](https://user-images.githubusercontent.com/50409210/139699132-65783fac-6657-42b5-a0b5-cbbaa38d6654.png)
![image](https://user-images.githubusercontent.com/50409210/139699291-e5d7de53-3819-44db-b817-05708bd8a367.png)

## Model Selection and Evaluation

- **Models Tested**:
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - XGBoost

- **Metrics**:
    - AUC Score
    - F1 Score
    - Classification Reports

- **Results**:
    - After evaluation, XGBoost was chosen for its best AUC score.

*Model comparison:*

![image](https://user-images.githubusercontent.com/50409210/139699504-2deae157-1877-4df0-b41b-887ac8923e79.png)

## Handling Class Imbalance

- Used SMOTE to oversample the minority class.
- Re-evaluated models with balanced classes; XGBoost continued to perform best.

## Deployment

### Local Deployment with Docker

1. **Exported the Trained Model**:
    - Converted the notebook code to a Python script (`FinalModeltrain.py`).
    - Saved the trained model using Pickle.

2. **Created a Flask Web Service**:
    - Developed `FinalModelpredict.py` to serve predictions via an API.

3. **Managed Dependencies with Pipenv**:
    - Created a `Pipfile` and `Pipfile.lock` to manage project dependencies.

4. **Built a Docker Container**:
    - Wrote a `Dockerfile` to containerize the application.
    - Built the Docker image using the Dockerfile.

5. **Testing the Model**:
    - Used `FinalModelpredicttest.py` to send sample inputs and receive predictions.

*Example output from Flask app:*

![image](https://user-images.githubusercontent.com/50409210/139581819-0fe4351e-f48f-4c2c-910d-4945506bf1ba.png)

### AWS Cloud Deployment using Elastic Beanstalk

1. **Set Up Elastic Beanstalk CLI**:
    - Installed the Elastic Beanstalk CLI using Pipenv.

2. **Initialized the Application**:
    - Ran `eb init` to set up the application for deployment.

3. **Deployed to AWS**:
    - Created an environment with `eb create`.
    - Deployed the Docker container to AWS Elastic Beanstalk.

4. **Testing in AWS**:
    - Used the provided AWS URL to send requests and receive predictions.

*Example AWS deployment and output:*

![image](https://user-images.githubusercontent.com/50409210/139911782-c07e2ca9-30f6-4f91-9f42-cb50216e14a5.png)
![image](https://user-images.githubusercontent.com/50409210/139920909-6616f3f1-af9e-4ae2-b879-f098b48ea3a0.png)

## Instructions for Local Deployment of Stroke Prediction Project

To deploy the **XGBoost for Classification** model locally as an app using Docker, follow these steps:

1. **Set Directory**:
    - Change to the desired directory in your command prompt.

2. **Train the Model**:
   ```bash
   python FinalModeltrain.py
   ```

3. **Install Flask**:
   ```bash
   pip install flask
   ```

4. **Run Flask App**:
   ```bash
   python FinalModelpredict.py
   ```

5. **Install Waitress (for Windows)**:
   ```bash
   pip install waitress
   ```

6. **Run Waitress Server**:
   ```bash
   waitress-serve --listen=127.0.0.1:5000 FinalModelpredict:app
   ```

7. **Test Model**:
   Open a new command prompt and run:
   ```bash
   python FinalModelpredicttest.py
   ```
   This will display stroke predictions for a new person.

8. **Set Up Virtual Environment with Pipenv**:
   ```bash
   pip install pipenv
   pipenv install numpy scikit-learn==0.24.2 flask pandas requests xgboost
   pipenv install --python 3.8
   pipenv shell
   ```

9. **Test Model in Virtual Environment**:
   ```bash
   python FinalModelpredicttest.py
   ```

10. **Docker Deployment**:
    - **Download Docker Image**:
      ```bash
      docker run -it --rm python:3.8.12-slim
      ```
    - **Build Docker Image**:
      ```bash
      docker build -t zoomcamp-test .
      ```
    - **Run Docker Image**:
      ```bash
      docker run -it --rm -p 5000:5000 zoomcamp-test
      ```
    This will launch the service on `http://0.0.0.0:5000`. Run predictions as before in a new terminal.

![image](https://user-images.githubusercontent.com/50409210/139918569-2944542e-6e11-41fd-add0-dd1f21bd6f30.png)

At last, our Stroke Prediction project has been deployed locally using a Docker container.

## Cloud Deployment of Stroke Prediction Service using AWS Elastic Beanstalk

Deploy the model to AWS Elastic Beanstalk with these steps:

1. **Install Elastic Beanstalk CLI**:
   ```bash
   pipenv install awsebcli --dev
   ```

2. **Enter Virtual Environment**:
   ```bash
   pipenv shell
   ```

3. **Initialize Elastic Beanstalk**:
   ```bash
   eb init -p docker stroke-serving
   ```

4. **Local Test**:
   ```bash
   eb local run --port 5000
   ```

5. **Make Predictions**:
   Run prediction script:
   ```bash
   python FinalModelpredicttest.py
   ```

![image](https://user-images.githubusercontent.com/50409210/139911782-c07e2ca9-30f6-4f91-9f42-cb50216e14a5.png)

6. **Create AWS Environment**:
   ```bash
   eb create stroke-serving-env
   ```
   This provides a public URL to access the application.

![image](https://user-images.githubusercontent.com/50409210/139920909-6616f3f1-af9e-4ae2-b879-f098b48ea3a0.png)

7. **Test Cloud Service**:
   Send a request to the URL to get predictions:
   ```bash
   python FinalModelpredicttest.py
   ```

It results in giving us stroke predictions and probabilities from our model for the new sample person (as shown below).

![image](https://user-images.githubusercontent.com/50409210/139912332-cf342652-d25d-499f-a57e-dd2b9831de0a.png)

Thus, our stroke-prediction service is deployed inside a container on AWS Elastic Beanstalk. To reach it, we use its public URL.

![image](https://user-images.githubusercontent.com/50409210/139912524-47a45d6b-15f0-4060-9c4f-6e8a6d672337.png)

## Conclusion

This project successfully built and deployed an XGBoost model to predict stroke risk, providing insights into stroke-related factors and an accessible web-based tool hosted on AWS.

---
