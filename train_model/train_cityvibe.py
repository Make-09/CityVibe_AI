import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Архитектура Deep Sets
class CityVibeNet(nn.Module):
    def __init__(self):
        super(CityVibeNet, self).__init__()
        
        # Phi: Обрабатывает один объект (тип и время)
        # Вход: [class_id, walk_time] -> Выход: вектор признаков (16 чисел)
        self.phi = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Rho: Берет сумму векторов объектов + NDVI и дает оценку
        # Вход: 16 (от объектов) + 1 (NDVI) = 17 -> Выход: 1 (score)
        self.rho = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Ограничиваем результат от 0 до 1
        )

    def forward(self, ndvi, objects):
        # objects имеет форму [batch, num_objects, 2]
        # Прогоняем каждый объект через Phi
        obj_embeddings = self.phi(objects) 
        
        # Суммируем признаки всех объектов (Pooling)
        summed_embeddings = torch.sum(obj_embeddings, dim=1)
        
        # Объединяем с NDVI
        combined = torch.cat([summed_embeddings, ndvi], dim=1)
        
        # Получаем финальный балл
        return self.rho(combined)

# 2. Загрузка данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        ndvi = torch.tensor([[item['input']['ndvi']]], dtype=torch.float32)
        target = torch.tensor([[item['target']]], dtype=torch.float32)
        
        # Превращаем объекты в тензор [N, 2]
        obj_list = [[obj['class'], obj['walk_time']] for obj in item['input']['objects']]
        # Добавляем "паддинг" или просто упаковываем в тензор (в нашем случае batch=1 для простоты)
        obj_tensor = torch.tensor([obj_list], dtype=torch.float32)
        
        formatted_data.append((ndvi, obj_tensor, target))
    return formatted_data

# 3. Обучение
def train():
    dataset = load_data('cityvibe_dataset.json')
    model = CityVibeNet()
    criterion = nn.MSELoss() # Среднеквадратичная ошибка
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Начинаем обучение...")
    for epoch in range(200): # 200 эпох
        epoch_loss = 0
        for ndvi, objects, target in dataset:
            optimizer.zero_grad()
            output = model(ndvi, objects)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch+1}, Ошибка: {epoch_loss/len(dataset):.4f}")

    # Сохраняем веса модели
    torch.save(model.state_dict(), "cityvibe_model.pth")
    print("✅ Модель сохранена как 'cityvibe_model.pth'")

if __name__ == "__main__":
    train()