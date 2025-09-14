# WasteDetectorWEB ♻️

WasteDetectorWEB is a **web-based intelligent waste classification system** built with **TensorFlow/Keras** for training and **Flask** for deployment.  
The project allows users to upload an image of waste, and the system will predict its category in real time.  

---

## ⚙️ Requirements

This project was developed with **Python 3.10.0** to ensure compatibility with TensorFlow and Keras.  

### 🛠️ Required Libraries

Install the following dependencies before running the project:

- **tensorflow / keras** → Deep learning framework for training and inference.  
- **flask** → Web framework to create the prediction API and web interface.  
- **opencv-python** → For image preprocessing.  
- **numpy** → Array and numerical data management.  
- **pillow (PIL)** → Image manipulation.  
- **scikit-learn** → Dataset splitting for training and testing.  
- **matplotlib** → For plotting training results.  

You can install them manually with:

```bash
pip install tensorflow flask opencv-python numpy pillow scikit-learn matplotlib
```

📥 Download / Clone the Repository

```bash
git clone https://github.com/xrl3y/WasteDetectorWEB.git
cd WasteDetectorWEB
```

📂 Project Structure

```bash
WasteDetectorWEB/
│── app.py              # Flask web app (API + frontend)
│── train_model.py      # Script to train the CNN model with TensorFlow/Keras
│── modelo_basura.h5    # Trained AI model (TensorFlow H5 format)
│── static/             # Static resources (CSS, JS, images)
│── templates/          # HTML templates (Flask frontend)
```

🔎 Explanation

- app.py → The main file to run the web server. Provides:

/ route → renders the homepage (templates/index.html).

/predict route → API endpoint to classify uploaded images.

- train_model.py → Script to train/retrain the waste classification model. Includes:

   - Dataset preprocessing.

  - CNN model definition.

  - Data augmentation.

  - Training loop with up to 100 epochs.

  - Model saved as modelo_basura.h5.

modelo_basura.h5 → Pre-trained model that classifies 5 categories:

- Metal

- Organic

- Paper & Cardboard

- Plastic

- Glass

static/ → Frontend assets (CSS, JavaScript, images).

templates/ → HTML files rendered by Flask (e.g., index.html).

🚀 Usage
Train (optional, only if you want a new model):

```bash
python train_model.py
```

This will train a CNN using the dataset and save the model as modelo_basura.h5.

Run the web application:

```bash
python app.py
```
Open your browser and go to:

```bash
http://127.0.0.1:5000/
```

Upload an image of waste, and the system will predict its category in real time.

📊 Model Training
The CNN is trained with the TrashNet dataset (trashnet/data/dataset-resized).

- Image size: 128x128 pixels.

- Batch size: 32.

- Epochs: 100.

- Data augmentation: rotation, shifts, flips.

The trained model is stored as:

```bash
modelo_basura.h5
```
During testing, the model reached an accuracy of ~85% depending on the dataset split.

🎨 Web Interface
The web app provides a simple form to upload images and see predictions.

Example screenshot:



<p align="center"> <img width="1252" height="809" alt="image" src="https://github.com/user-attachments/assets/ac648bb7-55a5-462f-bcd1-5bdb854e2cba" /> </p>

✅ Recommendations
Always use Python 3.10.0 for compatibility with TensorFlow.

- Train in a machine with a GPU if possible (training is much faster).

- Ensure your dataset has balanced classes for better accuracy.

- Keep the modelo_basura.h5 file updated when retraining.

- Modify the frontend (templates/ and static/) to adapt the look & feel.

## Author

This project was developed by **xrl3y**.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">



## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

