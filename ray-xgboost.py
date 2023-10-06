#!pip3 install -I git+https://github.com/cloudera/cmlextensions.git
#!pip3 install -I ray
#!pip3 install ray[client]
#!pip3 install ray[tune]
#!pip3 install xgboost_ray

import cmlextensions.ray_cluster as rc
import cmlapi
import os
import json
from pprint import pprint
import ray

# Set the setup variables needed by CML APIv2
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_APIV2_KEY")
PROJECT_NAME = os.getenv("CDSW_PROJECT")
PROJECT_ID=os.getenv("CDSW_PROJECT_ID")

cml = cmlapi.default_client(url=HOST,cml_api_key=API_KEY)

def set_environ(Cml,Item,Value):
    Project=Cml.get_project(os.getenv("CDSW_PROJECT_ID"))
    if Project.environment=='':
        Project_Environment={}
    else:
        Project_Environment=json.loads(Project.environment)
    Project_Environment[Item]=Value
    Project.environment=json.dumps(Project_Environment)
    Cml.update_project(Project,project_id=os.getenv("CDSW_PROJECT_ID"))

def get_environ(Cml,Item):
    Project=Cml.get_project(os.getenv("CDSW_PROJECT_ID"))
    Project_Environment=json.loads(Project.environment)
    return Project_Environment[Item]
  
cluster = rc.RayCluster( 
                        num_workers=2,
                         worker_cpu=4, worker_memory=8, worker_nvidia_gpu=2, 
                         head_cpu=4, head_memory=8, head_nvidia_gpu=0                       
                       )
cluster.init(360)
set_environ(cml,"RAY_ADDRESS", cluster.get_client_url())

cluster.ray_worker_details

runtime_env = {"RXGB_PLACEMENT_GROUP_TIMEOUT_S":"500"}

ray.init(address=cluster.get_client_url(),runtime_env=runtime_env)

from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import load_breast_cancer

train_x, train_y = load_breast_cancer(return_X_y=True)
train_set = RayDMatrix(train_x, train_y)

num_actors = 2
num_gpus_per_actor = 1
num_cpus_per_actor = 1

ray_params = RayParams(
    num_actors=num_actors, cpus_per_actor = num_cpus_per_actor, gpus_per_actor=num_gpus_per_actor)

evals_result = {}
bst = train(
    {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method":"hist", "device":"cuda"
        #"tree_method": "gpu_hist"
    },
    train_set,
    evals_result=evals_result,
    evals=[(train_set, "train")],
    verbose_eval=False,
    ray_params=ray_params)

bst.save_model("model.xgb")
print("Final training error: {:.4f}".format(
    evals_result["train"]["error"][-1]))

from xgboost_ray import RayDMatrix, RayParams, predict
from sklearn.datasets import load_breast_cancer
import xgboost as xgb

data, labels = load_breast_cancer(return_X_y=True)

dpred = RayDMatrix(data, labels)

bst = xgb.Booster(model_file="model.xgb")
pred_ray = predict(bst, dpred, ray_params=RayParams(num_actors=2))

print(pred_ray)

from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import load_breast_cancer

num_actors = 4
num_gpus_per_actor = 1
num_cpus_per_actor = 1

ray_params = RayParams(
    num_actors=num_actors, cpus_per_actor = num_cpus_per_actor, gpus_per_actor=num_gpus_per_actor)

def train_model(config):
    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)

    evals_result = {}
    bst = train(
        params=config,
        dtrain=train_set,
        evals_result=evals_result,
        evals=[(train_set, "train")],
        verbose_eval=False,
        ray_params=ray_params)
    bst.save_model("model.xgb")

from ray import tune

# Specify the hyperparameter search space.
config = {
    "tree_method": "approx",
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "eta": tune.loguniform(1e-4, 1e-1),
    "subsample": tune.uniform(0.5, 1.0),
    "max_depth": tune.randint(1, 9),
    "tree_method" : "gpu_hist"
}

# Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
analysis = tune.run(
    train_model,
    config=config,
    metric="train-error",
    mode="min",
    num_samples=4,
    resources_per_trial=ray_params.get_tune_resources())
print("Best hyperparameters", analysis.best_config)
    
ray.shutdown()
cluster.terminate()