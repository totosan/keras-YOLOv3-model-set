# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '69e5dabf-6be7-42fd-bb83-b066e03b4052'
resource_group = 'MVPSession'
workspace_name = 'ML-Workspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_id(workspace, 'a0e59ed1-c017-4c20-a722-0f2e5ea52db5')
dataset.download(target_path='./ds', overwrite=True)