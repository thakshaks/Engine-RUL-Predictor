import mlflow
from datetime import datetime
from evaluate import evaluate_predictions, nasa_score 

def train_and_log(model, model_name, X_tr, y_tr, X_v, y_v, params, is_dl=False):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        now = datetime.now()
        formatted_time1 = now.strftime("%H:%M:%S")
        if is_dl:
            model.fit(X_tr, y_tr, epochs=params.get("epochs", 15), batch_size=params.get("batch_size", 64), verbose=0)
            y_pred = model.predict(X_v).flatten()
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_v)
        
        rmse, mae, r2, nasa_score = evaluate_predictions(y_v, y_pred)

        now = datetime.now()
        formatted_time2 = now.strftime("%H:%M:%S")
        time_difference = datetime.strptime(formatted_time2, "%H:%M:%S") - datetime.strptime(formatted_time1, "%H:%M:%S")
        # mlflow.log_metric("val_rmse", rmse)
        # mlflow.log_metric("val_nasa_score", nasa_score)
        # mlflow.log_metric("val_mae", mae)
        # mlflow.log_metric("val_r2", r2)
        # mlflow.log_metric("training_time", time_difference.total_seconds())
        if is_dl:
            mlflow.keras.log_model(model, name=model_name)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_nasa_score", nasa_score)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_r2", r2)
            mlflow.log_metric("training_time", time_difference.total_seconds())
        else:
            mlflow.sklearn.log_model(
                model,
                name=model_name,
                skops_trusted_types=["sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
            )
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_nasa_score", nasa_score)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_r2", r2)
            mlflow.log_metric("training_time", time_difference.total_seconds())
        print(f"[{model_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.2f} | NASA Score: {nasa_score:.2f}")