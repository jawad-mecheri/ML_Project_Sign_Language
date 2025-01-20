# **ML_Project_Sign_Language**

## **Project Overview**

This project aims to develop a sign language recognition application using machine learning models. The goal is to transform input data into sign language predictions, making communication more accessible for individuals with hearing impairments.

---

## **Continuous Deployment of Machine Learning Models**

### **Overview**

This project involves deploying a predictive machine learning model into production with a focus on automation, performance monitoring, and continuous training. The solution incorporates API development, containerization, web-based interaction, and reporting tools to ensure seamless deployment and operation in a production environment.

### **Features**
- **Model Serving API**: A FastAPI-based service to make predictions using the trained ML model.
- **User Interface**: A Streamlit web application for data input and prediction visualization.
- **Reporting**: Automated performance monitoring using Evidently.
- **Continuous Deployment**: Automatic retraining and redeployment of models based on new production data.
- **Containerized Architecture**: Docker-based containers for consistent and reproducible deployments.

### **Project Components**
1. **Data Preparation**:
   - Input data is transformed into feature vectors using an embedding model and stored in `ref_data.csv`.
   - Reference data (`ref_data.csv`) is used for model training and serves as a baseline for drift detection.
   
2. **Model Training**:
   - A machine learning pipeline is developed, including preprocessing, embedding, and a predictive model.
   - Trained artifacts (model, embedding, and scaler) are saved as pickle files in the `artifacts` directory.
   
3. **API Development**:
   - A FastAPI-based serving API handles POST requests for predictions.
   - The API loads pre-trained artifacts on startup to ensure quick response times.
   - An additional endpoint for feedback collection is implemented to gather real-world performance data.
   
4. **Web Application**:
   - Developed with Streamlit, the interface allows users to upload data, receive predictions, and provide feedback.
   - Communicates with the serving API via RESTful endpoints.
   
5. **Performance Monitoring**:
   - Uses Evidently to generate metrics such as classification performance (e.g., F1-score, recall) and data drift analysis.
   - Static and interactive dashboards provide insights into the health of the model and data.
   
6. **Automation**:
   - Retraining of the model is triggered based on the accumulation of feedback data.
   - A monitoring system checks every 5 minutes for changes in `prod_data.csv`. If five new elements are added, the system automatically retrains the model.
   - All automation processes run within a Docker environment.

---

## **Project Structure**

### **Root Directory**
- **artifacts/**: Contains trained models and associated files:
  - `model_xgb.pkl`: Trained XGBoost model.
  - `pca.pkl` and `scaler.pkl`: Dimension reduction and data normalization objects.
- **data/**: Data used for training and testing the model:
  - `prod_data.csv`: Production data.
  - `ref_data.csv` and `test_data.csv`: Reference and test data.
- **reporting/**: Files related to report generation or container management.
  - `docker-compose.yml` and `Dockerfile`: Docker configurations.
- **scripts/**: Python scripts for data preprocessing and model training.
  - `preprocess_data.py`: Data preprocessing script.
  - `retrain_model.py`: Script to retrain the model.
  - `train_model_ML1.ipynb`: Jupyter Notebook for model training.
  - `transform_data_ML1.ipynb`: Notebook for data transformation.
- **serving/**: Files for deploying the model in production.
  - `api.py`: API for interacting with the model.
  - `docker-compose.yml` and `Dockerfile`: Docker configurations for deployment.
- **webapp/**: User interface or backend for accessing project functionalities.
- **README.md**: Project documentation.

---

## **Technologies Used**
- **Programming Language**: Python.
- **Frameworks and Libraries**:
  - Scikit-learn, XGBoost: For model development and training.
  - FastAPI: For backend API development.
  - Streamlit: For the web interface.
  - Evidently: For generating model health reports.
- **Tools and Platforms**:
  - Docker: For containerizing applications.
  - Jupyter Notebook: For model development, experimentation, and analysis.
  - Git: For version control.
  - VSCode: Integrated Development Environment.

---

## **Prerequisites**
- **Docker and Docker Compose**: Ensure Docker Desktop is installed. [Docker installation guide](https://docs.docker.com/desktop/).
- **Python 3.10**: Required for API and model-related scripts.

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd ML_Project_Sign_Language
   ```

2. **Install Dependencies:**
   Ensure `pip` is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Docker:**
   If using Docker to deploy the application, build the image:
   ```bash
   docker-compose build
   ```

---

## **Usage**

1. **Start the API:**
   If using Docker, start the application with:
   ```bash
   cd serving
   docker-compose up
   ```
   The API will be accessible via `http://localhost:8080`.

2. **Start the Web Application:**
   If using Docker, start the application with:
   ```bash
   cd webapp
   docker-compose up
   ```
   The web application will be accessible via `http://localhost:8081`.

3. **Start the Reporting:**
   If using Docker, start the application with:
   ```bash
   cd reporting
   docker-compose up
   ```
   The reporting will be accessible via `http://localhost:8082`.

---

## **Contributors**

- Djawad Mecheri.
- Lotfi Mayouf.
- Abdennour Slimani.
- Akram Mekbal.
- Massinissa Maouche.
- Ishak Bouchlagheme.

---

## **License**

This project is licensed under the MIT License.
