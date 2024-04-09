from sklearn.model_selection import train_test_split  # для разбиения данных
from model_preparation import data_preparation  # для подготовки данных
from sklearn.ensemble import RandomForestRegressor  # для обучения
import pickle  # для сохранения модели


clf = RandomForestRegressor(random_state=42)  # создаем модель

path = r'd:/Python/Environments/AutoML/MLOps1/train/train_data.csv'

df = data_preparation(path)  # подготовка данных

# разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df.drop(
    'Target_regression', axis=1), df['Target_regression'], test_size=0.2,
                                                    random_state=42)

clf.fit(X_train, y_train)  # обучение

pickle.dump(clf, open('AutoML/MLOps1/model.pkl', 'wb'))  # сохранение модели

print('Модель сохранена в AutoML/MLOps1/model.pkl')  # вывод
