import json
import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from numpy import asarray, expand_dims
from model.architecture import InceptionResNetV1

class FaceRecognition:
    def __init__(self, model_path: str):
        self._model = InceptionResNetV1(weights_path=model_path)
        self._embeddings_db = self._load_db()

    @staticmethod
    def _load_image(filename: str) -> np.ndarray:
        image = Image.open(filename)
        image = image.convert("RGB")
        return asarray(image)

    @staticmethod
    def _extract_face(image: np.ndarray, required_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        return asarray(face_image)

    def _get_embedding(self, face_pixels: np.ndarray) -> np.ndarray:
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = self._model.predict(samples)
        return yhat[0]

    def _process_image(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        image = self._load_image(filepath)
        face = self._extract_face(image)
        if face is None:
            print("Nenhum rosto detectado.")
            return None, None
        embedding = self._get_embedding(face)
        return face, embedding

    @staticmethod
    def _plot_face(face_array: np.ndarray) -> None:
        plt.imshow(face_array)
        plt.axis('off')
        plt.show()

    @staticmethod
    def _calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def add_to_db(self, image_path: str, name: str) -> None:
        face, embedding = self._process_image(image_path)
        if face is not None and embedding is not None:
            self._embeddings_db.append({'name': name, 'embedding': embedding.tolist()})
            self._save_db()
            print(f"Adicionado ao banco de dados: {name}")
        else:
            print("Falha ao processar a imagem para adicionar ao banco de dados.")

    def find_in_db(self, image_path: str, threshold: float = 0.5) -> Optional[str]:
        face, embedding = self._process_image(image_path)
        if face is None or embedding is None:
            print("Falha ao processar a imagem para comparação.")
            return None

        best_match = None
        best_similarity = -1

        for entry in self._embeddings_db:
            similarity = self._calculate_cosine_similarity(embedding, np.array(entry['embedding']))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry['name']

        if best_similarity >= threshold:
            print(f"Melhor correspondência: {best_match} com similaridade {best_similarity}")
        else:
            print("Nenhuma correspondência encontrada.")
        
        return best_match

    def _save_db(self, filename: str = 'embeddings_db.json') -> None:
        with open(filename, 'w') as f:
            json.dump(self._embeddings_db, f)

    def _load_db(self, filename: str = 'embeddings_db.json') -> List[dict]:
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

class RealTimeFaceRecognition(FaceRecognition):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def recognize_from_camera(self, threshold: float = 0.5) -> None:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            face = self._extract_face(frame)
            if face is not None:
                embedding = self._get_embedding(face)
                best_match = None
                best_similarity = -1

                for entry in self._embeddings_db:
                    similarity = self._calculate_cosine_similarity(embedding, np.array(entry['embedding']))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry['name']

                if best_similarity >= threshold:
                    cv2.putText(frame, f'{best_match} ({best_similarity:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Desconhecido', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Real-Time Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


