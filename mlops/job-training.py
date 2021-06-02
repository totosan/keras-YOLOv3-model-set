# description: train tensorflow NN model on mnist data

# imports
import os
import argparse
import subprocess
from pathlib import Path
from azureml.core import Workspace, Dataset
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.data import OutputFileDatasetConfig

if __name__ == "__main__":
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
        default="gpucluster",
        help="Name of the used compute system (cluster, instance,...) in Azure ML",
    )
    parser.add_argument(
        "--download_weights",
        default=False,
        action="store_true",
        help="Add this flag, if you need to download darnet pretrained model.",
    )

    
    FLAGS = parser.parse_args()
    
    
    
    # get workspace
    # ------------------------
    ws = Workspace.from_config()
    ds = ws.get_default_datastore()

    
    # Prepare Data: 
    #---------------------------
    
    # download weights & convert
    if FLAGS.download_weights:
        cmdDownloadWeights = f"wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        subprocess.call(cmdDownloadWeights, shell=True)
        cmdConvertWeights = f"python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5"
        subprocess.call(cmdConvertWeights, shell=True)
    
    print("Uploading weights...")
    ds.upload_files(["./weights/yolov4.h5"],target_path="Yolo4/Weights",overwrite=False, show_progress=True)


    output_data1 = OutputFileDatasetConfig(destination = (ds, 'outputdataset/{run-id}'))
    output_data_dataset = output_data1.register_on_complete(name = 'prepared_output_data')

    # get root of git repo
    prefix = Path(__file__).resolve().parents[1]

    # images from datastorage
    training_data_path = Dataset.File.from_files((ds, "YoloTraining/Data/Source_Images/Training_Images/vott-csv-export/"))  
    annotation_path = Dataset.File.from_files((ds,"YoloTraining/Data/Source_Images/Training_Images/vott-csv-export/*.csv"))
    
    # weights from datastorage
    weights_ds = Dataset.File.from_files((ds, "Yolo4/Weights/"))
    
    
    # Prepare environment
    #------------------------
    
    # environment file
    environment_file = str(prefix.joinpath("mlops/pip-env.yml"))

    # azure ml settings
    environment_name = FLAGS.env_name
    experiment_name = FLAGS.experiment_name
    compute_name = FLAGS.compute_name

    # create environment
    env = Environment.from_conda_specification(environment_name, environment_file)
    #env.environment_variables['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    env.environment_variables['TF_NEED_CUDA'] = '1'
    env.environment_variables['TF_NEED_TENSORRT'] = '1'
    env.environment_variables['TF_CUDA_COMPUTE_CAPABILITIES'] = '3.5,5.2,6.0,6.1,7.0'
    env.environment_variables['TF_CUDA_VERSION'] = '10.1'

    # Specify a GPU base image
    env.docker.enabled = True
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

    os.makedirs("./outputs", exist_ok=True)
    weightsPath = weights_ds.as_mount()
    args=[
            "--gpu_num",1,
            "--batch_size",16,
            "--val_split",0.1,
            "--decay_type", None, #default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant']
            "--transfer_epoch",20,
            "--total_epoch",50, #default 250
            "--model_type","yolo4_mobilenetv2_lite",
            "--anchors_path","configs/yolo4_anchors.txt",
            "--annotation_file",annotation_path.to_path()[0].strip("/"),
            "--classes_path","configs/custom_classes.txt",
            "--model_image_size","416x416",
            #"--weights_path", weightsPath,
            "--trainings_data_path",training_data_path.as_mount(),
            "--log_dir", "./outputs",
        ]
    
    run_id="yolo-4"

    # Do training
    # --------------------
    
    # prepare training script
    script_dir = str(prefix.joinpath("."))
    script_name = "train.py"

    # create job config
    src = ScriptRunConfig(
        source_directory=script_dir,
        script=script_name,
        environment=env,
        compute_target=compute_name,
        arguments=args
    )

    experiment = Experiment(ws, run_id, experiment_name)
    
    # submit training job
    run = experiment.submit(src)
    run.wait_for_completion(show_output=False)
 
    
    # register models (checkpoint, staged & final together)
    # ------------------------------------------------------
    model_name = "yolov4"    
    model = run.register_model(model_name=model_name,
                            tags={'area': 'Yolo'},
                            model_path='./outputs/trained_final.h5')
    print("Registered model:")
    print(model.name, model.id, model.version, sep='\t')

