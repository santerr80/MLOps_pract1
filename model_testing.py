import pickle  # для загрузки модели
from model_preparation import data_preparation  # для подготовки данных


def load_model(model_path):  # функция для загрузки модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


path = r'AutoML/MLOps1/test/test_data.csv'  # путь к тестовым данным

clf = load_model('AutoML/MLOps1/model.pkl')  # загружаем модель

df = data_preparation(path)  # подготовка данных

# разделяем данные на X и y
X_test, y_test = df.drop('Target_regression', axis=1), df['Target_regression']
y_pred = clf.predict(X_test)  # делаем предсказания

# выводим сообщение об успешном тестировании
print("Теститорвание прошло успешно:")

# запрос на сохранение предсказаний
save = input("Сохранить предсказания? (y/n): ")
if save == 'y':
    with open('AutoML/MLOps1/predictions.txt', 'w') as f:
        f.write(str(y_pred))  # сохраняем предсказания
