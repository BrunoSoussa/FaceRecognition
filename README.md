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
   ```sh
      git clone https://github.com/yourusername/facerecognition.git
      cd facerecognition
2. **Instalar os pacotes necessários**
   pip install numpy tensorflow opencv-python pillow matplotlib


3. **Uso**
   # Classe FaceRecognition
   A classe FaceRecognition fornece métodos para processar imagens, extrair embeddings faciais e gerenciar um banco de dados de embeddings.

   Inicialização
   python
   from facerecognition import FaceRecognition
   
   model_path = 'caminho/para/inceptionresnetv1_weights.h5'
   recognition = FaceRecognition(model_path)
   
   Adicionar um Rosto ao Banco de Dados
   python
   recognition.add_to_db('caminho/para/imagem.jpg', 'Nome da Pessoa')
   
   Encontrar um Rosto no Banco de Dados
   python
   result = recognition.find_in_db('caminho/para/imagem.jpg')
   print(result)

Classe RealTimeFaceRecognition
A classe RealTimeFaceRecognition estende FaceRecognition para fornecer reconhecimento facial em tempo real usando uma webcam.

Inicialização
from facerecognition import RealTimeFaceRecognition

model_path = 'caminho/para/inceptionresnetv1_weights.h5
real_time_recognition = RealTimeFaceRecognition(model_path)

Reconhecer Rostos da Webcam
real_time_recognition.recognize_from_camera()

Pressione q para sair do reconhecimento em tempo real.

Descrições Detalhadas dos Métodos

Classe FaceRecognition

- __init__(self, model_path: str)
  - Inicializa o modelo e carrega o banco de dados de embeddings.

- _load_image(filename: str) -> np.ndarray
  - Carrega uma imagem de um arquivo e converte para um array NumPy.

- _extract_face(image: np.ndarray, required_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]
  - Detecta e extrai o primeiro rosto de uma imagem, redimensionando para o tamanho necessário.

- _get_embedding(face_pixels: np.ndarray) -> np.ndarray
  - Gera um embedding para um rosto usando o modelo InceptionResNetV1.

- _process_image(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]
  - Processa uma imagem para extrair um rosto e gerar seu embedding.

- _plot_face(face_array: np.ndarray) -> None
  - Plota uma imagem de rosto usando Matplotlib.

- _calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float
  - Calcula a similaridade cosseno entre dois embeddings.

- add_to_db(self, image_path: str, name: str) -> None
  - Adiciona um rosto e seu embedding ao banco de dados.

- find_in_db(self, image_path: str, threshold: float = 0.5) -> Optional[str]
  - Encontra a melhor correspondência para um rosto no banco de dados com base na similaridade cosseno.

- _save_db(self, filename: str = 'embeddings_db.json') -> None
  - Salva o banco de dados de embeddings em um arquivo JSON.

- _load_db(self, filename: str = 'embeddings_db.json') -> List[dict]
  - Carrega o banco de dados de embeddings de um arquivo JSON.

Classe RealTimeFaceRecognition

- __init__(self, model_path: str)
  - Inicializa o modelo e carrega o banco de dados de embeddings, herdando de FaceRecognition.

- recognize_from_camera(self, threshold: float = 0.5) -> None
  - Captura frames de uma webcam e realiza reconhecimento facial em tempo real, exibindo os resultados na tela.


