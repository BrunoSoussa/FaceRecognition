# Sistema de Reconhecimento Facial

## Visão Geral

Este projeto oferece um sistema de reconhecimento facial utilizando um modelo pré-treinado Facenet. Ele inclui funcionalidades para adicionar novos rostos a um banco de dados, encontrar correspondências no banco de dados para uma imagem fornecida e reconhecimento facial em tempo real usando uma webcam.

## Dependencias

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- PIL (Pillow)
- Matplotlib
- JSON

## Instalção

1. **Clone o repositório:**
   ```python
      git clone https://github.com/BrunoSoussa/FaceRecognition.git
      cd facerecognition
2. **Instalar os pacotes necessários**
   ```sh
   pip install numpy tensorflow opencv-python pillow matplotlib



4. **Uso**
   ```python
   from model.prediction import RealTimeFaceRecognition
   
   if __name__ == "__main__":

       # Adicionar ao banco de dados exemplo 
       recognizer.add_to_db("exemplo.jpeg", "Nome Exemplo")
       
       recognizer = RealTimeFaceRecognition(model_path=r"model/keras/facenet_keras.h5")
       recognizer.recognize_from_camera()
   

