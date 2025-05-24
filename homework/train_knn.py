#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better

x_train, x_test, y_train, y_test = prepare_data(
    file_path="data/winequality-red.csv",
    test_size=0.25,
    random_state=123456,
)

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)

estimator.fit(x_train, y_train)

mse, mae, r2 = calculate_metrics(estimator, x_train, y_train)

print_metrics("Train metrics", mse, mae, r2)

# Metricas de error durante testing

mse, mae, r2 = calculate_metrics(estimator, x_test, y_test)

print_metrics("Test metrics", mse, mae, r2)

save_model_if_better(estimator, x_test, y_test)
# save_model_if_better(estimator, x_test, y_test)
