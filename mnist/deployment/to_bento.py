from MNIST_CNN import Network
import torch
import bentoml



def load_model_and_save_to_bento(path):
    model = Network()
    model.load_state_dict(torch.load(path))
    model.eval()

    bento_model = bentoml.pytorch.save_model('mnist_cnn', model, signatures={"__call__": {"batchable": True, "batch_dim": 0}})
    print(f'Bento model tag: {bento_model.tag}')

if __name__ == "__main__":

    load_model_and_save_to_bento('../model/model.pth')