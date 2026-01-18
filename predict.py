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
            nn.Linear(17, 32), # 16 от объектов + 1 от NDVI
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Твоя модель выдает 0...1
        )

    def forward(self, ndvi, objects):
        # objects: [1, N, 2]
        obj_embeddings = self.phi(objects) 
        summed_embeddings = torch.sum(obj_embeddings, dim=1)
        combined = torch.cat([summed_embeddings, ndvi], dim=1)
        return self.rho(combined)

device = torch.device("cpu")
model = CityVibeNet()
MODEL_PATH = "cityvibe_model.pth"

if os.path.exists(MODEL_PATH):
    # Убеждаемся, что загружаем именно ту структуру, которую обучали
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Модель Deep Sets загружена корректно")

def predict_score(ndvi, objects_list):
    """
    ndvi: float (например 0.606)
    objects_list: список [{'class': 1, 'walk_time': 5.0}, ...]
    """
    with torch.no_grad():
        # Подготовка NDVI [1, 1]
        ndvi_tensor = torch.tensor([[float(ndvi)]], dtype=torch.float32)
        
        # Подготовка объектов [1, N, 2]
        if not objects_list:
            obj_tensor = torch.zeros((1, 1, 2), dtype=torch.float32)
        else:
            objs = [[float(o['class']), float(o['walk_time'])] for o in objects_list]
            obj_tensor = torch.tensor([objs], dtype=torch.float32)
        
        # Получаем предсказание (0...1)
        prediction = model(ndvi_tensor, obj_tensor).item()
        
    # Переводим из 0...1 в 0...100 баллов
    return int(prediction * 100)