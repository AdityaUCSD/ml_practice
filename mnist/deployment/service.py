import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENTO_MODEL_TAG = 'mnist_cnn:latest'

classifier_runner = bentoml.pytorch.get(BENTO_MODEL_TAG).to_runner()
mnist_service = bentoml.Service("mnist_classifier", runners=[classifier_runner])

@mnist_service.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(input_data: np.ndarray) -> np.ndarray:
    batch_ret = await classifier_runner.async_run(input_data.reshape(1, 28, 28))
    return batch_ret.numpy()