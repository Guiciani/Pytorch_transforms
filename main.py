# As transformações nos dadasets são utilizadas para ajustar os dados para o 
# treinamento.
# Todos os datasets tem 2 parametros, -transform para modificar as features
# e target_transform para modificar as labels que aceitam os "chamaveis" que
# contem a logica da transformação

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


#Carregando os dados  do FashionMNIST

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


# ToTensor(), converte a PIL image ou Numpy array em FloatTensor. e escala
# os pixels da intensidade da imagem em valores no range [0,1]

#Lambda Transforms
# Aplica qualquer função lambda definida pelo dev. Aqui definimos uma função
# para tornar um inteiro em tensor encapsulado. Primeiro cria um tensor 0 de 
# tamanho 10 e chama o scatter_ que designa o value=1 no index efetuado 
# pelo label y

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

print(target_transform)