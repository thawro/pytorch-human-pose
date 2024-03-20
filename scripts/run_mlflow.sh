echo "...Starting MLFlow server..."

cd mlflow
source .mlflow_env
mlflow server --host ${HOST} --port ${PORT} --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${ARTIFACT_ROOT} --artifacts-destination ${ARTIFACT_ROOT}
