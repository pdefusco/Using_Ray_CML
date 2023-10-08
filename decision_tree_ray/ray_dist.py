!pip3 install -I git+https://github.com/cloudera/cmlextensions.git
!pip3 install pyarrow
!pip3 install ray[default]==2.2.0
#!pip3 install ray[client]
#!pip3 install ray[tune]
#!pip3 install xgboost_ray
!pip3 install tqdm

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
                         worker_cpu=4, worker_memory=8, worker_nvidia_gpu=0,
                         head_cpu=4, head_memory=8, head_nvidia_gpu=0
                       )
cluster.init(360)
set_environ(cml,"RAY_ADDRESS", cluster.get_client_url())

cluster.ray_worker_details

#runtime_env = {"RXGB_PLACEMENT_GROUP_TIMEOUT_S":"500"}

ray.init(address=cluster.get_client_url())


# EXAMPLE 1: RAY DATASETS
items = [{"name": str(i), "data": i} for i in range(10000)]
ds = ray.data.from_items(items)
ds.show(5)

squares = ds.map(lambda x: x["data"] ** 2)

evens = squares.filter(lambda x: x % 2 == 0)
evens.count()

cubes = evens.flat_map(lambda x: [x, x**3])
sample = cubes.take(10)
print(sample)

"""The drawback of Dataset transformations is that each step gets executed synchronously.
In this example that is a nonissue, but for complex tasks that, for example,
mix reading files and processing data, you would want an execution
that can overlap individual tasks.
DatasetPipeline does exactly that. Let’s rewrite the previous example into a pipeline:"""

pipe = ds.window()

result = pipe\
    .map(lambda x: x["data"] ** 2)\
    .filter(lambda x: x % 2 == 0)\
    .flat_map(lambda x: [x, x**3])
result.show(10)

@ray.remote
def retrieve_task(item):
    return retrieve(item)

start = time.time()
object_references = [
    retrieve_task.remote(item) for item in range(8)
]
data = ray.get(object_references)
print_runtime(data, start)



@ray.remote
def remote_hi():
    import os
    import socket
    return f"Running on {socket.gethostname()} in pid {os.getpid()}"
future = remote_hi.remote()
ray.get(future)


import timeit

def slow_task(x):
    import time
    time.sleep(2) # Do something sciency/business
    return x

@ray.remote
def remote_task(x):
    return slow_task(x)

things = range(10)

very_slow_result = map(slow_task, things)
slowish_result = map(lambda x: remote_task.remote(x), things)

slow_time = timeit.timeit(lambda: list(very_slow_result), number=1)
fast_time = timeit.timeit(lambda: list(ray.get(list(slowish_result))), number=1)
print(f"In sequence {slow_time}, in parallel {fast_time}")



@ray.remote
def crawl(url, depth=0, maxdepth=1, maxlinks=4):
    links = []
    link_futures = []
    import requests
    from bs4 import BeautifulSoup
    try:
        f = requests.get(url)
        links += [(url, f.text)]
        if (depth > maxdepth):
            return links # base case
        soup = BeautifulSoup(f.text, 'html.parser')
        c = 0
        for link in soup.find_all('a'):
            try:
                c = c + 1
                link_futures += [crawl.remote(link["href"], depth=(depth+1),
                                   maxdepth=maxdepth)]
                # Don't branch too much; we're still in local mode and the web is big
                if c > maxlinks:
                    break
            except:
                pass
        for r in ray.get(link_futures):
            links += r
        return links
    except requests.exceptions.InvalidSchema:
        return [] # Skip nonweb links
    except requests.exceptions.MissingSchema:
        return [] # Skip nonweb links

ray.get(crawl.remote("http://holdenkarau.com/"))

"""
The actor model can be used for everything from real-world systems like email,
to Internet of Things (IoT) applications like tracking temperature,
to flight booking. A common use case for Ray actors is managing state (e.g., weights)
while performing distributed ML without requiring expensive locking.2

Ray actors are created and called similarly to remote functions
but use Python classes, which gives the actor a place to store state.
You can see this in action by modifying the classic “Hello World” example
to greet you in sequence
"""

@ray.remote
class HelloWorld(object):
    def __init__(self):
        self.value = 0
    def greet(self):
        self.value += 1
        return f"Hi user #{self.value}"

# Make an instance of the actor
hello_actor = HelloWorld.remote()

# Call the actor
print(ray.get(hello_actor.greet.remote()))
print(ray.get(hello_actor.greet.remote()))
