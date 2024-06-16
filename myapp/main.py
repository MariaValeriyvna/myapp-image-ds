import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()

# docs - это swagger
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    try:
         # Преобразуем строку в массив numpy
        image_str = image.strip()[1:-1]  # Убираем квадратные скобки
        image_array = np.array(list(map(int, image_str.split(',')))).reshape(28, 28)
        pred = model.predict(image_array)

        # Преобразуем предсказание в строку, если это необходимо
        if isinstance(pred, np.generic):
            pred = pred.item()  # Преобразуем numpy тип данных в стандартный Python тип

        return {'prediction': pred}
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {'error': str(e)}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
