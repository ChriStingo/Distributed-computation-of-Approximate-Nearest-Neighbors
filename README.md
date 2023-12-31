# **Distributed computation of Approximate Nearest Neighbors**
This project references to my master thesis work "Distributed computation of Approximate Nearest Neighbors" for my Master's Degree in Computer Science in external collaboration with [Wikimedia Foundation](https://wikimediafoundation.org/). The work focuses on making the image upload service on [Wikipedia](https://en.wikipedia.org/wiki/Main_Page) more efficient, studying how to distribute the computation of Approximate Nearest Neighbors, developing ad hoc solutions, exploring existing possibilities and adapting them to perform distributed searches of similar images within the Wikimedia dataset.

## **What is this repository**
This repository is intended to contain implementations of several libraries for performing similarity (cosine) searches in order to find the best one for my purpose. The various implementations are based on the previously exposed use case.

### **Approximate Nearest Neighbors**
 - [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html): this library is not approximated and is highly inefficient for my research use case, its implementation, with sklearn, has been provided only for comparison purposes with approximated libraries. Inside KNN you will also find a brute-force version that execute a linear comparison between a single vector and the entire dataset, without creating a data structure.
 - [Annoy](https://github.com/spotify/annoy): this library implements a version of ANN, written by Erik Bernhardsson on behalf of Spotify.
 - [Faiss](https://github.com/facebookresearch/faiss): this library was created for efficient similarity searches and clustering of dense vectors by the facebook (now Meta) research team.
 - [SPTAG](https://github.com/microsoft/SPTAG): this library is designed for large scale vector approximate nearest neighbor search, created by Microsoft as an alternative version to their previous library, called DiskANN. Unlike other technologies, this solution involves the use of RAM memory and the physical disk, efficiently managing which and how much information save on them.

### **Distribuited Approximate Nearest Neighbors**
- [Distributed-Faiss](https://github.com/facebookresearch/distributed-faiss): this library has already been covered, but not its distributed version. This algorithm, although very efficient, does not add much new to the proposals previously analyzed that runs on a single machine. What makes this library special, however, is that it can be distributed on multiple machines in a very simple way.
- [Distributed-SPTAG](https://github.com/microsoft/SPTAG): this library has already been covered, but not its distributed version. This algorithm, although very efficient, does not add much new to the proposals previously analyzed that runs on a single machine. What makes this library special, however, is that it can be distributed on multiple machines in a very simple way.
- [Milvus](https://github.com/milvus-io/milvus): this library implements vector database that focuses on providing an efficient system and architecture for similarity searches and AI-powered applications. The need that Milvus tries to satisfy is to provide a complete system capable of operating on huge dynamic datasets, providing solutions that can also run on GPUs also in a distributed way. To do this Milvus was build on libraries that have already been explained as Faiss, making the necessary changes for its use case.

## **How it is structured**
The project has been structured in a precise way to ensure easy organization of files and to make navigation within it simple.

 - [Algorithms](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Algorithms "Algorithms"): inside this folder you can find all the implementations
	 - for-each-library:
		 - config.py: this file contains the configurations necessary for the correct operation of the libraries
		 - init.py: this file contains the code for reading datasets, training data structures and creating the index
		 - search.py: this file contains the code for performing searches and returning the resulting items
     		 - other files: chronometer.py, mocks.py
 - [Datasets](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets "Datasets"): the datasets and data structures saved in memory will be inserted inside this folder
	 - [Compressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Compressed): in this folder it is possible to insert the compressed datasets, as can be seen from the example inserted. you may also notice a file called [links.txt](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/Compressed/links.txt) containing all Wikimedia datasets
	 - [Decompressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Decompressed): in this folder it is possible to insert the decompressed datasets, from here they will be filtered and decomposed for a correct functioning of the libraries
	 - [Images](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Images): inside this folder it is possible to find the images corresponding to the vectors extracted from the dataset
	 - [Vectors](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Vectors): within this folder it is possible to find the vectors corresponding to the images extracted from the dataset
	 - [formatDatasets.bash](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/formatDatasets.bash): this file, executable following this [README.md](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/README.md), contains the code to decompose all the datasets from the Decompressed folder moving the vectors inside the Vectors/ folder and the corresponding images inside the Images/ folder. This file uses [formatDatasets.py](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/formatDatasets.py) to convert .csv file into binary format to speed up all the project.
  - [Docs](https://github.com/ChriStingo/Distributed-computation-of-Approximate-Nearest-Neighbors/tree/main/Docs "Docs"): documents obtained during the writing of the thesis

## **How to execute**

### **Download some datasets**

 1. Go to [Compressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Compressed) folder and execute `wget -i links.txt` to download the Wikimedia dataset ([images](https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/image_pixels/), [vectors](https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/resnet_embeddings/))
 2. Decompress archives into [Decompressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Decompressed), in Wikimedia dataset case `gunzip *.gz`
 3. Go into [Datasets](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets "Datasets") folder and follow the [README.md](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/README.md)

### **Import / Install / Compile the libraries**
First go into [Algorithms](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Algorithms "Algorithms") folder and install general requirements with `pip install -r requirements.txt`. 
 - KNN: `pip install scikit-learn` or don't install anything if you prefer to use the brute-force version
 - Annoy: `pip install annoy`  
 - Faiss: follow the [conda tutorial](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
 - SPTAG: you can [compile](https://github.com/microsoft/SPTAG/blob/main/README.md) this library via source code, or use Docker. I personally recommend the latter solution, by running these commands:
	```bash
 	set GIT_LFS_SKIP_SMUDGE=1 
	git clone --recurse-submodules https://github.com/microsoft/SPTAG
 	cd SPTAG
	docker build -t sptag .
	```
- Distributed-Faiss: clone the [repository](https://github.com/facebookresearch/distributed-faiss) (if you want to use my PR for adding precision to search phase with `set_index_parameter` or `set_index_parameters` methods clone this [repository]([https://github.com/facebookresearch/distributed-faiss](https://github.com/ChriStingo/distributed-faiss/tree/set-index-parameter))), enter the folder and then run `pip install -e .`
- Distributed-SPTAG: follow the instruction for *SPTAG*
- Milvus: `pip install pymilvus` and then milvus can be install with [docker compose](https://milvus.io/docs/install_standalone-docker.md) or [Kubernetes](https://milvus.io/docs/install_standalone-helm.md) (Milvus Standalone for testing) or with [Helm + Kubernetes](https://milvus.io/docs/install_cluster-helm.md) (Milvus Cluster for production)

### **Execute the ANN libraries**

Before running the libraries customize the various config.py files
1. `python init.py`
2. `python search.py`

### **Execute the Distributed-ANN libraries**
Before running the libraries customize the various config.py files
- Distributed-Faiss: 
	1. Customize the [DISCOVERY_CONFIG.txt]('https://github.com/ChriStingo/Distributed-computation-of-Approximate-Nearest-Neighbors/blob/main/Algorithms/Distributed-Faiss/DISCOVERY_CONFIG.txt') file.
  	2. Execute on each server `python init_server.py`
  	3. Execute on the client `python init_client.py`
  	4. Execute on the client `python search.py`
- Distributed-SPTAG:
  	1. Go to each server, customize the config.py file (attention to START_METADATA_OFFSET) and run `python init.py`
	1. Go to the servers hosting the previously created index. In the Release folder of the library, import and customize the [service.ini](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Algorithms/Distributed-Sptag/service.ini "service.ini") file and run the command `./server -m socket -c service.ini`
	2. Go to the machine you identify as the aggregator. In the Release folder of the library, import and customize the [Aggregator.ini](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Algorithms/Distributed-Sptag/Aggregator.ini "Aggregator.ini") file and run the command `./aggregator`
	3. Go to the client, customize the config.py file, and execute `python search.py`
- Milvus:
  	1. `python init.py`
	2. `python search.py`


---
Made by Christian Stingone
