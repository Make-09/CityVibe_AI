import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Архитектура Deep Sets
class CityVibeNet(nn.Module):
    def __init__(self):
        super(CityVibeNet, self).__init__()
        
        # phi: обработка одного объекта (категория и время)
        self.phi = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # rho: агрегация объектов + NDVI и получение итогового score
        self.rho = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # ограничение результата в диапазоне [0, 1]
        )

    def forward(self, ndvi, objects):
        # objects: [batch, num_objects, 2]
        obj_embeddings = self.phi(objects) 
        
        # суммирование признаков объектов (pooling)
        summed_embeddings = torch.sum(obj_embeddings, dim=1)
        
        # объединение с NDVI
        combined = torch.cat([summed_embeddings, ndvi], dim=1)
        
        # итоговый score
        return self.rho(combined)

# загрузка данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        ndvi = torch.tensor([[item['input']['ndvi']]], dtype=torch.float32)
        target = torch.tensor([[item['target']]], dtype=torch.float32)
        
        # objects -> tensor [N, 2]
        obj_list = [[obj['class'], obj['walk_time']] for obj in item['input']['objects']]
        # batch=1
        obj_tensor = torch.tensor([obj_list], dtype=torch.float32)
        
        formatted_data.append((ndvi, obj_tensor, target))
    return formatted_data

# обучение
def train():
    dataset = load_data('cityvibe_dataset.json')
    model = CityVibeNet()
    criterion = nn.MSELoss()  # среднеквадратичная ошибка
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    for epoch in range(200):  # 200 эпох
        epoch_loss = 0
        for ndvi, objects, target in dataset:
            optimizer.zero_grad()
            output = model(ndvi, objects)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Error: {epoch_loss/len(dataset):.4f}")

    # сохранение весов модели
    torch.save(model.state_dict(), "cityvibe_model.pth")
    print("Model saved: cityvibe_model.pth")

if __name__ == "__main__":
    train()