# Test-Time-Adaption
This repo contains code from two repositories - 1. [TTA for Blind Image Quality Assessment](https://github.com/subhadeeproy2000/TTA-IQA) and 2.[Depth-Anything](https://github.com/LiheYoung/Depth-Anything) \
The two repositories have been merged only for MetaIQA and for KONIQ dataset inside TTA folder (``` TTA/TTA-IQA/MetaIQA ```) <br>
### Datasets
Detailed description and link of datasets is given in the [TTA-IQA](https://github.com/subhadeeproy2000/TTA-IQA) repo. 
Each dataset needs to be prepared according to the ```folders.py``` inside TTA folder (```TTA/TTA-IQA/MetaIQA/folders.py```)
### Pre-trained models
Both the IQA and Depth-Anything pre-trained models are available in their respective repositories.
### Code files
Every TTA model's folder has a ```ALL_EXPT.py``` file that lists the experiments and arguments the author pases while testing.
```MetaIQA``` folder has been updated by integrating the above mentioned repositories. The following briefly explains each code file :
* ```dataloader.py``` and ```folders.py``` : prepares the dataset by extracting the MOS code, image paths and defined the pre-processing. <b>NOTE:</b> Dataset needs to be prepared according to how it is being used in the ```folders.py``` file. For example, the datasets may have images, MOS code and other information stored in separate files which maybe requied to be extracted and filled in a single ```.csv``` file which must be named exactly as given in ```folders.py``` file.
* ```sam.py``` : this file has been taken from [this repo](https://github.com/davda54/sam). Simply copy this file in each model's folder to use it.
* ```ttt.py``` : this is the main file of ```MetaIQA``` folder. It includes all the arguments and some arguments have been added such as each loss' weights and choice of optimizer (either Adam of SAM)
* ```depth-anything``` : this folder is taken from [Depth-Anything]([url](https://github.com/LiheYoung/Depth-Anything)) repo and is the only file which is required to run their model. It has been imported in ```ttt.py``` file .
