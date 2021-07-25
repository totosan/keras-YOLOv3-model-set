import argparse, os
from pathlib import Path
from azureml.core.run import Run

    
def registerModel(register_name, model_path, project_name, run_context):    
    toPath = os.path.join('./outputs', model_path)
    finetuning_model = run_context.register_model(  model_name  = register_name,
                                                    model_path  = toPath,
                                                    tags        = {'area': 'Yolo'},
                                                    description = f"registerd for {project_name}")
    
def uploadArtefacts(model_file, input_path, run_context):
    files = Path(input_path)

    print(f"Listing folder '{input_path}'")
    for file in files.iterdir():
        print(f"    {file.name}")
    
    fromFullPath = os.path.join(input_path, model_file)
    toPath = os.path.join('./outputs', model_file)
    run_context.upload_file(name=toPath ,path_or_stream=fromFullPath)
    
def main(args):        
    run_context = Run.get_context()
    uploadArtefacts(model_file=args.model_path, input_path=args.input_dataset, run_context=run_context)
    registerModel(register_name=args.register_name, model_path=args.model_path, project_name=args.project_name, run_context=run_context)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument("--register_name",type=str,default=None,help="Name of the model under what to register",)
    parser.add_argument("--model_path",type=str,default=None,help="Path, where model file can be found for registration",)
    parser.add_argument("--project_name",type=str,default=None,help="Name of project for description of registered model",)
    parser.add_argument("--input_dataset",type=str,help="dataset, that is output of previous step with trained model artefact")
    args = parser.parse_args()
    
    main(args)    