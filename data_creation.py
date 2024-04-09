import pandas as pd
import numpy as np
from pathlib import Path

# Установите случайное зерно для воспроизводимости
np.random.seed(42)

loc, scale = 10, 1

# Сгенерируем синтетические данные
n_samples = 10000
features = {
    "Feature1": np.random.logistic(loc, scale, n_samples),
    "Feature2": np.random.poisson(20, n_samples),
    "Feature3": np.random.rand(n_samples),
    "Feature4": np.random.rand(n_samples),
    "Feature5": np.random.rand(n_samples),
}

# Вычислим регрессионную целевую переменную
features["Target_regression"] = (
    2 * features["Feature1"]
    + 3 * features["Feature2"]
    + 0.5 * features["Feature3"]
    + np.random.normal(loc=0, scale=0.1, size=n_samples)
)

# Создадим DataFrame
df_train = pd.DataFrame(features).sample(frac=0.9, random_state=42)
df_test = pd.DataFrame(features).drop(df_train.index)

# Укажим директорию для сохранения файла тренировочных данных
filepath_train = Path('AutoML/MLOps1/train/train_data.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)

# Укажим директорию для сохранения файла тестовых данных
filepath_test = Path('AutoML/MLOps1/test/test_data.csv')
filepath_test.parent.mkdir(parents=True, exist_ok=True)

# Сохраним данные в файлы CSV
df_train.to_csv(filepath_train, index=False)
df_test.to_csv(filepath_test, index=False)


print("Синтетические данные сохранены")
print("Файл тренировочных данных:", filepath_train)
print("Файл тестовых данных:", filepath_test)
