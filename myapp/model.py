import os
import pickle
from pathlib import Path

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        # your code here

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        # Инициализация словаря кодов и символов
        self.label_mapping = self.load_label_mapping()

    def load_label_mapping(self):
        label_mapping = {}
        mapping_file = Path(__file__).parent / '../emnist-balanced-mapping.txt'
        with open(mapping_file, 'r') as file:
            for line in file:
                label, ascii_code = map(int, line.strip().split())
                label_mapping[label] = chr(ascii_code)
        return label_mapping

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here

        x = x.reshape(1, -1)
        pred_code = self.model.predict(x)[0]
        pred_symbol = self.label_mapping.get(pred_code, '?') # Возвращает '?' если код не найден
        return (f"symbol: {pred_symbol}, code: {pred_code}")
