# Brain Tumor Classification API

A deep learning-based API for classifying brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

This project uses a custom-trained **ResNet-50** PyTorch model to perform the inference. The API is built using **Flask** and generates Grad-CAM overlays to visualize the model's attention.

## Features
- **Image Classification:** Fast inference using PyTorch's ResNet-50 architecture.
- **Grad-CAM Visualizations:** See exactly which parts of the MRI the model focused on when making its prediction.
- **REST API:** Simple `/predict` endpoint that accepts an image and returns JSON with predictions, confidences, and a base64 encoded heatmap.

## Tech Stack
- **Backend Framework:** Flask
- **Machine Learning:** PyTorch, Torchvision
- **Image Processing:** OpenCV (Headless), Pillow
- **Deployment:** Render (configured via `render.yaml`)

## Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask API:
   ```bash
   python app.py
   ```
4. The server will start on `http://127.0.0.1:5000/`. You can check the status via `GET /status`.

## Deployment
This repository includes a `render.yaml` file, which makes it ready to be deployed as a **Render Web Service** out-of-the-box.
1. Push this repository to GitHub.
2. Go to the [Render Dashboard](https://dashboard.render.com).
3. Create a new **Blueprint** and connect your repository.
4. Render will automatically build and deploy the API using Gunicorn.

## API Usage
**Endpoint:** `POST /predict`
**Form Data:** Send an image file under the key `image`.

**Example Response:**
```json
{
  "prediction": "Glioma",
  "prediction_index": 0,
  "confidence": 98.45,
  "all_confidences": {
    "Glioma": 98.45,
    "Meningioma": 0.50,
    "No Tumor": 0.05,
    "Pituitary": 1.00
  },
  "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

## Note on Inference
This system is a deep learning research prototype intended to support educational and exploratory analysis. It is not a diagnostic device and must not replace expert radiologic interpretation.
