# Hypertension Prediction API & Streamlit UI

A complete solution for hypertension risk prediction using a machine learning model, with both a FastAPI backend and a modern Streamlit web UI. The model uses a scikit-learn pipeline with StandardScaler and LogisticRegression, achieving approximately 85.5% accuracy.

---

## Features

- **FastAPI Backend**: REST API for single and batch predictions, model info, and health check
- **Streamlit UI**: User-friendly web interface for predictions, batch uploads, feature documentation, and result visualization
- **Docker Support**: Deploy both API and UI with a single Docker image
- **Prediction History**: See all predictions made in your session
- **Visualizations**: Probability bars and batch histograms for prediction confidence
- **Comprehensive Feature Documentation**: Sidebar explains all input features

---

## Model Features (Input Variables)

| Feature  | Type  | Description                                                                                                    |
| -------- | ----- | -------------------------------------------------------------------------------------------------------------- |
| age      | float | Age of the patient (years)                                                                                     |
| sex      | float | Sex (0 = Female, 1 = Male)                                                                                     |
| cp       | int   | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)              |
| trestbps | int   | Resting blood pressure (mm Hg)                                                                                 |
| chol     | int   | Serum cholesterol (mg/dl)                                                                                      |
| fbs      | int   | Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)                                                              |
| restecg  | int   | Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) |
| thalach  | int   | Maximum heart rate achieved                                                                                    |
| exang    | int   | Exercise induced angina (1 = Yes, 0 = No)                                                                      |
| oldpeak  | float | ST depression induced by exercise relative to rest                                                             |
| slope    | int   | Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)                               |
| ca       | int   | Number of major vessels colored by fluoroscopy (0-4)                                                           |
| thal     | int   | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect, 0 = unknown)                                 |

---

## Project Structure

```
lpmh/
├── main.py                    # FastAPI application
├── streamlit_app.py           # Streamlit web UI
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── logistic_pipeline_model.pkl  # Trained model
├── logistic_pipeline_model_hypertension.ipynb  # Model training notebook
├── test_api.py                # Automated API test script
├── example_client.py          # Example API client
├── Dockerfile                 # Docker build file (API & UI)
├── docker-compose.yml         # Docker Compose setup
├── deploy.sh                  # Deployment helper script
└── LICENSE
```

---

## Running the API and UI

### 1. **Locally (Recommended for Development)**

#### Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

#### Start the FastAPI backend:

```bash
python main.py
```

API will be at [http://localhost:8000](http://localhost:8000)

#### Start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

UI will be at [http://localhost:8501](http://localhost:8501)

---

### 2. **Docker (Production/Deployment)**

#### Build the image:

```bash
docker build -t hypertension-app .
```

#### Run the API (default):

```bash
docker run -p 8000:8000 hypertension-app
```

#### Run the Streamlit UI:

```bash
docker run -p 8501:8501 -e APP_MODE=ui hypertension-app
```

#### Or use Docker Compose:

```bash
docker-compose up
```

---

## Streamlit UI Features

- **Single Patient Prediction**: Enter all 13 features, get prediction, probability, and confidence, with a probability progress bar.
- **Batch Prediction**: Upload a CSV file for up to 100 patients, see results in a table and a probability histogram.
- **Prediction History**: All predictions (single and batch) in your session are saved and viewable in expandable sections.
- **Feature Documentation**: Sidebar expander explains each input variable.
- **Model Info & Health Check**: Sidebar buttons for model details and API health.
- **Modern, clean layout** with error handling and example CSV format.

---

## FastAPI Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/model-info` - Model details
- **POST** `/predict` - Single prediction
- **POST** `/predict-batch` - Batch predictions (max 100)

---

## Testing the API

### Using `test_api.py`

A comprehensive test script is provided to verify the API endpoints and error handling.

**Run all tests:**

```bash
python test_api.py
```

This script will:

- Check health and model info endpoints
- Test single and batch predictions
- Test invalid input handling

### Using `example_client.py`

A simple example client is provided to demonstrate how to interact with the API programmatically.

**Run the example client:**

```bash
python example_client.py
```

This script will:

- Check API health
- Retrieve model info
- Make a single prediction
- Make a batch prediction

---

## Visualizations & History

- **Single Prediction**: Probability is shown as a progress bar for intuitive confidence feedback.
- **Batch Prediction**: Probability distribution is shown as a bar chart.
- **History**: All predictions in the current session are saved and viewable in the UI.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
