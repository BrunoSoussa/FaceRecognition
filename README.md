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
      git clone https://github.com/yourusername/facerecognition.git
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

         # A classe RealTimeFaceRecognition estende FaceRecognition para fornecer reconhecimento facial em tempo real usando uma webcam.
         
         # Inicialização
         from facerecognition import RealTimeFaceRecognition
         
         model_path = 'caminho/para/inceptionresnetv1_weights.h5
         real_time_recognition = RealTimeFaceRecognition(model_path)
         
         # Reconhecer Rostos da Webcam
         real_time_recognition.recognize_from_camera()
   
