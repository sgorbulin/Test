import argparse
import sys
import os
import boto3
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient
from pandas import DataFrame
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


class Defaults:
    instance_type = 'ml.m4.xlarge'
    endpoint_name = 'xgboost-churn-predictions'
    AWS_region = 'eu-west-2'
    execution_role = 'execution_role_from_AWS_Account'
    instance_type = 'ml.m4.xlarge'
    model_params = {
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'max_depth': 10,
        'eta': 0.2,
        'gamma': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'num_round': 100
    }
    source_data_path = 's3://sagemaker-sample-files/datasets/tabular/synthetic/churn.txt'
    mlflow_tracking_uri = 'http://mlflow.prefix.dns-prefix.local'
    mlflow_experiment_name = 'adp-churn-prediction-exp'
    mlflow_registered_model_name = 'adp-churn-predictions'
    ecr_image_name = 'ecr_image_name/from_AWS_Account'
    sagemaker_bucket = 'sagemaker_bucket_from_AWS_Account'
    cloud_map_service_id = 'cloud_map_service_id_from_AWS_Account'
    cloud_map_instance_name = 'cloud_map_instance_id_from_AWS_Account'
    cloud_map_attribute_name = 'artifacts_uri'


def checking_existence_model(client, model_name):
    filter_string = f"name='{model_name}'"
    models = client.search_registered_models(filter_string=filter_string)
    return models


def get_production_model_version(client, model_name: str):
    model: RegisteredModel = client.get_registered_model(model_name)
    versions: List[ModelVersion] = [v for v in model.latest_versions
                                    if v.current_stage == "Production"]
    if versions:
        return versions[0]
    else:
        return None


def preprocess_data() -> tuple:
    log.info(f'Uploading data from {Defaults.source_data_path}')
    churn_df = pd.read_csv(Defaults.source_data_path)
    churn_df = churn_df.drop(["Phone", "Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)
    churn_df["Area Code"] = churn_df["Area Code"].astype(object)
    model_data = pd.get_dummies(churn_df)
    model_data = pd.concat(
        [model_data["Churn?_True."], model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)], axis=1
    )
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )
    log.info(f'Finished preprocessing, saving data...')
    return (train_data.drop(columns=['Churn?_True.']), train_data[['Churn?_True.']],
            validation_data.drop(columns=['Churn?_True.']), validation_data[['Churn?_True.']])


def train(train_x: DataFrame, train_y: DataFrame) -> XGBClassifier:
    model = XGBClassifier(**Defaults.model_params)
    xgb = model.fit(train_x, train_y.values.ravel())
    return xgb


def deploy_to_sagemaker(model_uri: str):
    aws_id = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = boto3.client('iam').get_role(RoleName=Defaults.execution_role)['Role']['Arn']
    image_url = aws_id + '.dkr.ecr.' + Defaults.AWS_region + '.amazonaws.com/' + Defaults.ecr_image_name + ':' + mlflow.version.VERSION
    mfs.deploy(
        Defaults.endpoint_name,
        model_uri=model_uri,
        region_name=Defaults.AWS_region,
        mode=mfs.DEPLOYMENT_MODE_REPLACE,
        execution_role_arn=role_arn,
        image_url=image_url,
        bucket=Defaults.sagemaker_bucket,
        instance_type=Defaults.instance_type
    )


def deploy_to_kubernetes():
    cloud_map = boto3.client('servicediscovery', region_name=Defaults.AWS_region)
    current_attributes = cloud_map.get_instance(ServiceId=Defaults.cloud_map_service_id,
                                                InstanceId=Defaults.cloud_map_instance_name)['Instance']['Attributes']
    current_attributes[Defaults.cloud_map_attribute_name] = artifact_uri
    cloud_map.register_instance(ServiceId=Defaults.cloud_map_service_id, InstanceId=Defaults.cloud_map_instance_name,
                                Attributes=current_attributes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--AWS_region', default=Defaults.AWS_region)
    parser.add_argument('--source_data_path', default=Defaults.source_data_path)
    parser.add_argument('--endpoint_name', default=Defaults.endpoint_name)
    parser.add_argument('--max_depth', default=Defaults.model_params['max_depth'])
    parser.add_argument('--gamma', default=Defaults.model_params['gamma'])
    parser.add_argument('--eta', default=Defaults.model_params['eta'])
    parser.add_argument('--min_child_weight', default=Defaults.model_params['min_child_weight'])
    parser.add_argument('--num_round', default=Defaults.model_params['num_round'])
    parser.add_argument('--mlflow_tracking_uri', default=Defaults.mlflow_tracking_uri)
    parser.add_argument('--mlflow_experiment_name', default=Defaults.mlflow_experiment_name)
    parser.add_argument('--mlflow_registered_model_name', default=Defaults.mlflow_registered_model_name)
    parser.add_argument('--instance_type', default=Defaults.instance_type)
    parser.add_argument('--execution_role', default=Defaults.execution_role)
    parser.add_argument('--ecr_image_name', default=Defaults.ecr_image_name)
    parser.add_argument('--sagemaker_bucket', default=Defaults.sagemaker_bucket)
    parser.add_argument('--cloud_map_service_id', default=Defaults.cloud_map_service_id)
    parser.add_argument('--cloud_map_instance_name', default=Defaults.cloud_map_instance_name)
    parser.add_argument('--cloud_map_attribute_name', default=Defaults.cloud_map_attribute_name)

    args = parser.parse_args()

    Defaults.model_params['max_depth'] = args.max_depth
    Defaults.model_params['gamma'] = args.gamma
    Defaults.model_params['eta'] = args.eta
    Defaults.model_params['min_child_weight'] = args.min_child_weight
    Defaults.model_params['num_round'] = args.num_round
    Defaults.mlflow_tracking_uri = args.mlflow_tracking_uri
    Defaults.mlflow_experiment_name = args.mlflow_experiment_name
    Defaults.mlflow_registered_model_name = args.mlflow_registered_model_name
    Defaults.AWS_region = args.AWS_region
    Defaults.source_data_path = args.source_data_path
    Defaults.endpoint_name = args.endpoint_name
    Defaults.instance_type = args.instance_type
    Defaults.execution_role = args.execution_role
    Defaults.ecr_image_name = args.ecr_image_name
    Defaults.sagemaker_bucket = args.sagemaker_bucket
    Defaults.cloud_map_service_id = args.cloud_map_service_id
    Defaults.cloud_map_instance_name = args.cloud_map_instance_name
    Defaults.cloud_map_attribute_name = args.cloud_map_attribute_name

    mlflow.set_tracking_uri(Defaults.mlflow_tracking_uri)
    mlflow.set_experiment(Defaults.mlflow_experiment_name)
    log.info(f'Ml flow version: {mlflow.version.VERSION}')
    with mlflow.start_run() as run:
        train_x, train_y, validate_x, validate_y = preprocess_data()
        xgb = train(train_x, train_y)
        predictions = xgb.predict(validate_x)
        predictions_proba = xgb.predict_proba(validate_x)
        tn, fp, fn, tp = confusion_matrix(validate_y, predictions).ravel()
        p = (tp + tn) / (tp + tn + fp + fn) * 100
        log.info("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
        log.info("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Churn", "Churn"))
        log.info("Observed")
        log.info("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Churn", tn / (tn + fn) * 100, tn,
                                                                      fp / (tp + fp) * 100, fp))
        log.info("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Churn", fn / (tn + fn) * 100, fn,
                                                                         tp / (tp + fp) * 100, tp))
        scores = cross_val_score(xgb, validate_x, validate_y, cv=5)
        accuracy = round(scores.mean() * 100, 2)

        mlflow_client = MlflowClient(Defaults.mlflow_tracking_uri)
        mlflow.set_experiment(Defaults.mlflow_experiment_name)
        mlflow.xgboost.log_model(xgb_model=xgb,
                                 artifact_path=Defaults.mlflow_experiment_name)
        if not checking_existence_model(mlflow_client, Defaults.mlflow_registered_model_name):
            mlflow_client.create_registered_model(Defaults.mlflow_registered_model_name)
        run_path = f'runs:/{run.info.run_id}/{Defaults.mlflow_experiment_name}'
        model_version = mlflow_client \
            .create_model_version(name=Defaults.mlflow_registered_model_name,
                                  source=run_path,
                                  run_id=run.info.run_id)
        prod_version = get_production_model_version(mlflow_client, Defaults.mlflow_registered_model_name)

        if prod_version:
            mlflow_client.transition_model_version_stage(name=Defaults.mlflow_registered_model_name,
                                                         version=prod_version.version,
                                                         stage='Archived')
        mlflow_client.transition_model_version_stage(name=Defaults.mlflow_registered_model_name,
                                                     version=model_version.version,
                                                     stage='Production')
        mlflow.log_params(Defaults.model_params)
        mlflow.log_metrics({'accuracy': p})
        artifact_uri = mlflow.get_artifact_uri()
        run_id = run.info.run_uuid
        experiment = mlflow_client.get_experiment_by_name(Defaults.mlflow_experiment_name)
        model_uri = os.path.join(artifact_uri, Defaults.mlflow_experiment_name)
        log.info(f'Run id = {run_id}; artifact_uri = {artifact_uri}; experimentId = {experiment.experiment_id}')
        if os.getenv('USE_SAGEMAKER_MODEL') and os.getenv('USE_SAGEMAKER_MODEL').lower() == 'enabled':
            deploy_to_sagemaker(model_uri)
        elif os.getenv('USE_KUBERNETES_MODEL') and os.getenv('USE_KUBERNETES_MODEL').lower() == 'enabled':
            deploy_to_kubernetes()
        else:
            deploy_to_sagemaker(model_uri)
            deploy_to_kubernetes()
