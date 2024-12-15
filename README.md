# Spark + Hadoop Cluster Environment (OUTDATED!)

# Recommended Reading

Please refer to the following tutorials before setting up Hadoop:
    
    - National Research Platform UserDocs (https://docs.nationalresearchplatform.org/userdocs/start/quickstart/):
        
        - Introduction:
        
            - Quickstart
        
        - Tutorials:
            
            - Docker and containers
            
            - Basic Kubernetes
    
    - Apache Hadoop Docs (https://hadoop.apache.org/docs/current/)
            
Usage of the NRP is not strictly necessary for this set up to work; deployment of these Pods will follow a similar process across other providers. If you wish to use these services, 
please refer to their 'Policies' (https://docs.nationalresearchplatform.org/userdocs/start/policies/) and 'Get Access' 
(https://docs.nationalresearchplatform.org/userdocs/start/get-access/) pages.
            
# Included

A primary image is used to create each Hadoop container; its image may be pulled here: 'gitlab-registry.nrp-nautilus.io/focegueda/hadoop-test/pysparktorch' . All the configurations and Dockerfile for image building is located in /Docker-Build/

/Deployments/ includes .yaml file configurations that start up each component separately:

    - service.yaml: Defines headless 'hadoop' service that exposes required ports for container communication.

    - ingress.yaml: Forwards WebUI components
    
    - pv.yaml: Defines a Persistent Volume in 'ReadWriteMany' to be shared between Pods. Do not use if you do not have control over the creation of Volumes.
    
    - pvc.yaml: Defines a Persistent Volume Claim in 'ReadWriteMany'to be shared between pods. This creates a temporary volume inside of the node. Nodes may configured to automatically
      claim storage from a remote destination; see the 'storageClassName' attribute inside the file. Any other storage options must use 'ReadWriteMany' in order to be shared between
      Pods; see https://docs.nationalresearchplatform.org/userdocs/storage/intro/.
      
    - master-pod.yaml: Defines the 'master' Hadoop container instance. This is where the primary NameNode is created and configured. It uses Bash commands to perform the initialization, so
      feel free to modify them.
      
    - slave01-pod.yaml: Defines the 'slave01' Hadoop container instance. This is where the secondary NameNode located.
    
    - slave02-pod.yaml: Defines the 'slave02' Hadoop container instance.

/Codebase/: includes files for testing distributed environment. Make sure to copy the MNIST folder to /user/MNIST in HDFS.
    
# Setting Up Pods

1. Ensure that kubectl is configured and the namespace is set. You may need to use 'export KUBECONFIG=$HOME/.kube/config.yaml' for kubectl to work properly.
    
2. Navigate to ' hadoop-test/Kubernetes-Yaml-Files/ ' directory in your command line.

3. Use ' kubectl apply -f example.yaml ' to apply the configuration to Kubernetes. This command can use multiple files at once. A quick option would use:
    
    - ' kubectl apply -f service.yaml -f pvc.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f master-pod.yaml '
    
4. Use ' kubectl get pods ' to see the status of your pod deployment. Also, use ' kubectl get pvcs' and ' kubectl get services ' to check the status of the other components.

    - If you are seeing problems with you deployment, you may use ' kubectl describe {component}' to check for any errors.

5. If each component is working properly, you may access the master container's shell with ' kubectl exec -it master -- /bin/bash/ '.

    - This master server can access the other containers through SSH with ' ssh {hostname} '. For example, you may connect to each container with ' ssh slave01 ' and ' ssh slave02 '.

6. Type ' jps ' in the command line to see that all the Hadoop Components are running, then ' hdfs dfsadmin -report ' to verify each component is connected.

7. Use ' exit ' to leave the container shell.

# Shutting Down Pods

Shutting down pods can be done with ' kubectl delete -f example.yaml ', and can also be chained like the ' apply ' command. Be aware that deleting your Pods & PVC will cause all data to be lost due to the stateless nature of Pods. For simplicity sake, use 'bash activate.sh' and 'bash deactivate.sh' to quickly deploy and destroy the pods.

# Other Acknowledgments

These containers are freely open within a namespace; please be careful where you are deploying your Pods. If you plan on deploying within a public namespace, recommendation would be to
pull the latest image from the repository, create a user account within the image, and modify the ssh passkeys. There is no easy way to do this during setup as of now.
