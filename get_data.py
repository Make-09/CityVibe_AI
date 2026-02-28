# main.py
from NDVI import get_ndvi_multi_radius
from infrastructure import get_cityvibe_input_data, CATEGORIES_MAP

# соответствие названий категорий их индексам (для входа в модель)
CLASS_TO_INT = {cls: i for i, cls in enumerate(CATEGORIES_MAP.keys())}

def get_cityvibe_for_nn(lat, lon):
    # NDVI
    ndvi_results = get_ndvi_multi_radius(lat, lon)
    ndvi_value = ndvi_results[200]  # основной радиус — 200 м

    # инфраструктура (с учетом NDVI)
    dataset = get_cityvibe_input_data(lat, lon, ndvi_value)

    # преобразование инфраструктуры в формат входа модели
    objects = []
    for item in dataset['infrastructure']:
        for cls_str, walk_time in item.items():
            cls_int = CLASS_TO_INT.get(cls_str, -1)
            if cls_int == -1:  # неизвестная категория
                continue
            objects.append({
                "class": cls_int,
                "walk_time": walk_time
            })

    # итоговый словарь для модели
    return {
        "ndvi": ndvi_value,
        "objects": objects
    }

if __name__ == "__main__":
    lat, lon = 51.09046123649118, 71.42606138571799
    data_for_nn = get_cityvibe_for_nn(lat, lon)

    print("\n--- CITYVIBE DATA FOR NN ---")
    print(data_for_nn)
