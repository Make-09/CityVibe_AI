import torch
import torch.nn as nn
import os

class CityVibeNet(nn.Module):
    def __init__(self):
        super(CityVibeNet, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.rho = nn.Sequential(
            nn.Linear(17, 32),  # 16 признаков от объектов + 1 признак NDVI
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # результат в диапазоне [0, 1]
        )

    def forward(self, ndvi, objects):
        # objects: [1, N, 2]
        obj_embeddings = self.phi(objects) 
        summed_embeddings = torch.sum(obj_embeddings, dim=1)
        combined = torch.cat([summed_embeddings, ndvi], dim=1)
        return self.rho(combined)

device = torch.device("cpu")

# Модель инициализируется в server.py, здесь только класс и функция predict_score
def predict_score(model, ndvi, objects_list):
    """
    model: экземпляр CityVibeNet
    ndvi: float (например 0.606)
    objects_list: список [{'class': 1, 'walk_time': 5.0}, ...]
    """
    with torch.no_grad():
        # NDVI -> tensor [1, 1]
        ndvi_tensor = torch.tensor([[float(ndvi)]], dtype=torch.float32)
        
        # objects -> tensor [1, N, 2]
        if not objects_list:
            obj_tensor = torch.zeros((1, 1, 2), dtype=torch.float32)
        else:
            objs = [[float(o['class']), float(o['walk_time'])] for o in objects_list]
            obj_tensor = torch.tensor([objs], dtype=torch.float32)
        
        # prediction in [0, 1]
        prediction = model(ndvi_tensor, obj_tensor).item()
        
    # преобразование в шкалу 0..100
    return int(prediction * 100)