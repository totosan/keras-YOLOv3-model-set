from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Datastore, Dataset
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.abspath("./utils/"))
from attach_compute import get_compute
from workspace import get_workspace

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
"""
Command line options
"""
parser.add_argument(
    "--env_name",
    type=str,
    default="TomowsEnv",
    help="The name of the training environment in Azure ML",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="yolo4_tuning",
    help="The name of the training environment in Azure ML",
)
parser.add_argument(
    "--compute_name",
    type=str,
    default="gpuinstancett",
    help="Name of the used compute system (cluster, instance,...) in Azure ML",
)
parser.add_argument(
    "--download_weights",
    default=False,
    action="store_true",
    help="Add this flag, if you need to download darnet pretrained model.",
)


FLAGS = parser.parse_args()


def main():
    load_dotenv()
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP_NAME")
    vm_size = os.environ.get("VM_SIZE")
    compute = os.environ.get("COMPUTE")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    model_name = os.environ.get("MODEL_NAME")
    ckpt_path = os.environ.get("MODEL_CHECKPOINT_PATH")
    build_id = os.environ.get("BUILD_BUILDID")
    pipeline_name = os.environ.get("TRAINING_PIPELINE_NAME")

    aml_workspace = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    print(aml_workspace)

    aml_compute = get_compute(
        aml_workspace,
        compute,
        vm_size)
    if aml_compute is not None:
        print(aml_compute)

    # getting datastorage
    ds = aml_workspace.get_default_datastore()
    training_data_path = Dataset.File.from_files((ds, "YoloTraining/Data/Source_Images/Training_Images/vott-csv-export/"))
    annotation_path = Dataset.File.from_files((ds, "YoloTraining/Data/Source_Images/Training_Images/vott-csv-export/*.csv"))
    outputs_path = PipelineData('outputs', is_directory=True, datastore=ds)

    # weights from datastorage
    weights_ds = Dataset.File.from_files((ds, "Yolo4/Weights/"))

    prefix = Path(__file__).resolve().parents[1]
    # environment file
    environment_file = str(prefix.joinpath("mlops/pip-env.yml"))

    run_config = RunConfiguration(conda_dependencies=CondaDependencies(conda_dependencies_file_path=environment_file))
    run_config.environment.docker.enabled = True
    run_config.environment.docker.gpu_support = True
    run_config.environment.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

    # Pipeline parameter
    model_name = PipelineParameter(name="model_name", default_value=model_name)
    release_id = PipelineParameter(name="release_id", default_value=build_id)
    total_epoch = PipelineParameter(name="total_epoch", default_value=50)

    #os.makedirs("./outputs", exist_ok=True)
    annotationPath = annotation_path.to_path()[0].strip("/")
    weightsPath = weights_ds.as_mount()
    sourceDir = str(prefix.joinpath("."))

    test_image_step = PythonScriptStep(
        name="Testing images",
        script_name="mlops/readImage.py",
        compute_target=aml_compute,
        source_directory=sourceDir,
        arguments=[
            "--csv_file",annotationPath,
            "--path",training_data_path.as_mount()
        ],
        runconfig=run_config,
        allow_reuse=False
    )
    print("Step for Debugging created")
    
    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        compute_target=aml_compute,
        source_directory=sourceDir,
        arguments=[
            # model
            "--model_type", "yolo4_mobilenetv2_lite",
            "--anchors_path", "configs/yolo4_anchors.txt",
            "--annotation_file", annotationPath,
            "--classes_path", "configs/custom_classes.txt",
            # compute
            "--gpu_num", 1,
            "--batch_size", 16,
            "--val_split", 0.1,
            "--model_image_size", "416x416",
            # training
            # default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant']
            "--decay_type", 'cosine',
            "--transfer_epoch", 20,
            "--total_epoch", total_epoch,
            # data
            #"--weights_path", weightsPath,
            "--trainings_data_path", training_data_path.as_named_input('Images').as_mount(),
            "--log_dir", outputs_path,
            "--eval_online"
        ],
        outputs= [outputs_path],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Step Train created")

    evaluate_step = PythonScriptStep(
        name="Test Model",
        script_name="mlops/utils/register_model.py",
        compute_target=aml_compute,
        source_directory=sourceDir,
        arguments=[
            "--model_file", "trained_final.h5",
            "--model_path", outputs_path,
            "--anchors_path", "./configs/yolo4_anchors.txt",
            "--classes_path", "./configs/custom_classes.txt",
            "--model_image_size", "416x416",
            "--eval_typ", "VOC",
            "--annotation_file", annotationPath,
            "--test_path", training_data_path.as_named_input('Images').as_mount(),
        ],
        inputs=[outputs_path],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Step Evaluate created")

    #train_step.run_after(test_image_step)
    #valuate_step.run_after(train_step)
    steps = [#test_image_step,
             train_step
             #, evaluate_step
             ]
    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline.validate()
    run = train_pipeline.submit(experiment_name="yolo-4", pipeline_parameters={"total_epoch":20})
    run.wait_for_completion(raise_on_error=True)
    
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="Model training/retraining pipeline",
        version=build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()
