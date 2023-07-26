import json
import numpy as np
import requests
import matplotlib.pyplot as plt
from torch import tensor

from training import load_and_split_dataset

SERVICE_URL = 'http://localhost:3000/predict'

def get_random_mnist_data_point(data_file='../DeepLearningPython/mnist.pkl.gz'):
    _, _, test_loader = load_and_split_dataset(data_file)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    random_index = np.random.randint(0, len(example_data))

    plt.imshow(example_data[random_index], cmap='gray', interpolation='none')
    plt.savefig('../figures/test_example.png')
    
    return example_data[random_index].numpy(), example_targets[random_index].numpy()
    

def make_request_to_bento_service(service_url, input_data):
    serialized = json.dumps(input_data.tolist())
    response = requests.post(
        service_url,
        data=serialized,
        headers={'content-type': 'application/json'}
    )
    return response.text

def main():
    input_data, expected_output = get_random_mnist_data_point()
    pred = make_request_to_bento_service(SERVICE_URL, input_data)
    pred = np.array(eval(pred)).argmax()
    print(f'Prediction: {pred}')
    print(f'Expected: {expected_output}')

if __name__ == '__main__':
    main()

