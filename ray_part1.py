from cmlextensions.ray_cluster import RayCluster

# Cluster Parameters (head_cpu,head_memory,num_workers,worker_memory,
# worker_cpu, env, dashboard_port) 
cluster = RayCluster(num_workers=2)
cluster.init()