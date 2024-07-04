# Face Recognition System

## Overview

This project provides a system for facial recognition using a pre-trained InceptionResNetV1 model. It includes functionality for adding new faces to a database, finding matches in the database for a given image, and real-time face recognition using a webcam.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- PIL (Pillow)
- Matplotlib
- JSON

## Installation

1. **Clone the repository:**
   ```python
      git clone https://github.com/yourusername/facerecognition.git
      cd facerecognition
2. **Instalar os pacotes necessários**
   ```sh
   pip install numpy tensorflow opencv-python pillow matplotlib



4. **Uso**
   # Classe FaceRecognition
   A classe FaceRecognition fornece métodos para processar imagens, extrair embeddings faciais e gerenciar um banco de dados de embeddings.
   Inicialização
    ```python
      from facerecognition import FaceRecognition
      
      model_path = 'caminho/para/inceptionresnetv1_weights.h5'
      recognition = FaceRecognition(model_path)
      
      # Adicionar um Rosto ao Banco de Dados
      recognition.add_to_db('caminho/para/imagem.jpg', 'Nome da Pessoa')
      
      # Encontrar um Rosto no Banco de Dados
      result = recognition.find_in_db('caminho/para/imagem.jpg')
      print(result)
  

# Classe RealTimeFaceRecognition
```python
   # A classe RealTimeFaceRecognition estende FaceRecognition para fornecer reconhecimento facial em tempo real usando uma webcam.
   
   # Inicialização
   from facerecognition import RealTimeFaceRecognition
   
   model_path = 'caminho/para/inceptionresnetv1_weights.h5
   real_time_recognition = RealTimeFaceRecognition(model_path)
   
   # Reconhecer Rostos da Webcam
   real_time_recognition.recognize_from_camera()


