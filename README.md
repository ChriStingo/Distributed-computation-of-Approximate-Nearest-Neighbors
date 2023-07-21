# **Distributed computation of Approximate Nearest Neighbors**
This project references to my master thesis work "Distributed computation of Approximate Nearest Neighbors" for my Master's Degree in Computer Science in external collaboration with [Wikimedia](https://www.wikimedia.org/). The work focuses on making the image upload service on [Wikipedia](https://en.wikipedia.org/wiki/Main_Page) more efficient, studying how to distribute the computation of Approximate Nearest Neighbors, developing ad hoc solutions, exploring existing possibilities and adapting them to perform distributed searches of similar images within the Wikimedia dataset.

## **What is this repository**
This repository is intended to contain implementations of several libraries for performing similarity (cosine) searches in order to find the best one for my purpose. The various implementations are based on the previously exposed use case.

### **Approximate Nearest Neighbors**
 - [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html): this library is not approximated and is highly inefficient for my research use case, its implementation has been provided only for comparison purposes with approximated libraries
 - [Annoy](https://github.com/spotify/annoy): this library implements a version of ANN, written by Erik Bernhardsson on behalf of Spotify
 - [FAISS](https://github.com/facebookresearch/faiss): this library was created for efficient similarity searches and clustering of dense vectors by the facebook (now Meta) research team.
 - [SPTAG](https://github.com/microsoft/SPTAG): this library is designed for large scale vector approximate nearest neighbor search, created by Microsoft as an alternative version to their previous library, called DiskANN. Unlike other technologies, this solution involves the use of RAM memory and the physical disk, efficiently managing which and how much information save on them.

### **Distribuited Approximate Nearest Neighbors**
- [Distributed-SPTAG](https://github.com/microsoft/SPTAG): this library has already been covered, but not its distributed version. This algorithm, although very efficient, does not add much new to the proposals previously analyzed that runs on a single machine. What makes this library special, however, is that it can be distributed on multiple machines in a very simple way.
- [Milvus](https://github.com/milvus-io/milvus): this library implements vector database that focuses on providing an efficient system and architecture for similarity searches and AI-powered applications. The need that Milvus tries to satisfy is to provide a complete system capable of operating on huge dynamic datasets, providing solutions that can also run on GPUs also in a distributed way. To do this Milvus was build on libraries that have already been explained as FAISS, making the necessary changes for its use case.
- [Elasticsearch](https://github.com/elastic/elasticsearch): *work in progress*

## **How it is structured**
The project has been structured in a precise way to ensure easy organization of files and to make navigation within it simple.

 - [Algorithms](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Algorithms "Algorithms"): inside this folder you can find all the implementations
	 - for-each-library:
		 - config.py: this file contains the configurations necessary for the correct operation of the libraries
		 - init.py: this file contains the code for reading datasets, training data structures and creating the index
		 - search.py: this file contains the code for performing searches and returning the resulting items
 - [Datasets](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets "Datasets"): the datasets and data structures saved in memory will be inserted inside this folder
	 - [Compressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Compressed): in this folder it is possible to insert the compressed datasets, as can be seen from the example inserted. you may also notice a file called [links.txt](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/Compressed/links.txt) containing all Wikimedia datasets
	 - [Decompressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Decompressed): in this folder it is possible to insert the decompressed datasets, from here they will be filtered and decomposed for a correct functioning of the libraries
	 - [Images](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Images): inside this folder it is possible to find the images corresponding to the vectors extracted from the dataset
	 - [Vectors](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Vectors): within this folder it is possible to find the vectors corresponding to the images extracted from the dataset
	 - [formatDatasets.bash](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/formatDatasets.bash): this file, executable following this [README.md](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/README.md), contains the code to decompose all the datasets from the Decompressed folder moving the vectors inside the Vectors/ folder and the corresponding images inside the Images/ folder.

## **How to execute**
### **Import / Install / Compile the libraries**
 - KNN: `pip install sklearn`
 - Annoy: `pip install annoy`  
 - FAISS: follow the [conda tutorial](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
 - SPTAG: you can [compile](https://github.com/microsoft/SPTAG/blob/main/README.md) this library via source code, or use Docker. I personally recommend the latter solution, by running these commands:
	```bash
	 set GIT_LFS_SKIP_SMUDGE=1 
	 git clone --recurse-submodules https://github.com/microsoft/SPTAG
	 docker build -t sptag .
	 ```
- Milvus: *work in progress*
- Elasticsearch: *work in progress*

### **Download some datasets**

 1. Go to [Compressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Compressed) folder and execute `wget -i links.txt` to download the Wikimedia dataset ([images](https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/image_pixels/), [vectors](https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/resnet_embeddings/))
 2. Decompress archives into [Decompressed](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets/Decompressed), in Wikimedia dataset case `gunzip *.gz`
 3. Go into [Datasets](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/tree/main/Datasets "Datasets") folder and follow the [README.md](https://github.com/ChriStingo/Approximate-Nearest-Neighbors-Searches/blob/main/Datasets/README.md)
 
### **Execute the libraries**
Before running the libraries customize the various config.py files
1. `python init.py`
2. `python search.py`

---
Made by Christian Stingone
