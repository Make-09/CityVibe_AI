# main.py
from NDVI import get_ndvi_multi_radius
from infrastructure import get_cityvibe_input_data, CATEGORIES

# создаем mapping категорий в int для deep sets
CLASS_TO_INT = {cls: i for i, cls in enumerate(CATEGORIES.keys())}

def get_cityvibe_for_nn(lat, lon):
    # 1️⃣ Получаем NDVI
    ndvi_results = get_ndvi_multi_radius(lat, lon)
    ndvi_value = ndvi_results[200]  # берем радиус 200 м как основной

    # 2️⃣ Получаем инфраструктуру
    dataset = get_cityvibe_input_data(lat, lon, ndvi_value)

    # 3️⃣ Преобразуем инфраструктуру в формат для NN
    objects = []
    for item in dataset['infrastructure']:
        for cls_str, walk_time in item.items():
            cls_int = CLASS_TO_INT.get(cls_str, -1)
            if cls_int == -1:  # на всякий случай
                continue
            objects.append({
                "class": cls_int,
                "walk_time": walk_time
            })

    # 4️⃣ Итоговый объект для нейросети
    return {
        "ndvi": ndvi_value,
        "objects": objects
    }

if __name__ == "__main__":
    lat, lon = 51.09046123649118, 71.42606138571799
    data_for_nn = get_cityvibe_for_nn(lat, lon)

    print("\n--- CITYVIBE DATA FOR NN ---")
    print(data_for_nn)
