## Knowledge Base Population Using Word2vec

The Knowledge Base Population (KBP) is to automatically identify 
relevant entities, learn and disc discover attributes about 
the its relations, and finally search, expand the KB with other relations. 

The idea is take a small set of samples pairs. 
Automatically defining semantic relation and expand the set
with new pairs.

## Installation

#### 1. Prerequisites

You need to have these libraries.

* *Python >= 3.0 * 
* *[gensim](https://radimrehurek.com/gensim/)* library
* *NumPy* and *SciPy*


#### 2. Setup 

Create environment folder.

```bash
virtualenv -p python3 env
```

Activate python environment.

```bash
source env/bin/activate
```

Install packages using requirements.txt file.

```bash
pip install -i requirements.txt
```

#### 3. Setting paths

In *config.py* file set paths:

* **word2vec_file** - Path to file with word embeddings dataset. 
Yo could be use any format also by word2vec (vec or bin) or custom vectors from gensim library. 
Popular pre-trained datasets can be found on official 
[word2vec page](https://code.google.com/archive/p/word2vec/) as [Google News dataset](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) (1.5GB).

* **output_file** - New expand set of pairs (entities have a possible semantic relation) whitout tag.
 
## Usage

Method can be accesed through command line by invoking *main.py* file:

```
usage: run_cli.py   [-h] [-o ouput_file] [-t num_results] 
                    [-m {1,2,3}][-n NEIGHBORHOOD] 
                    [-d {1,2}] [-s {1,2,3}]
                    input_file
```

Input file is a text file where there is one pairs of exemplary words on each line (examples are in *relations* folder in this repo), e.g.:

```
France Paris
Germany Berlin
Spain Madrid
```

The options are:

```
positional arguments:
  sample_file      filename of seed set file

optional arguments:
  -h, --help       show this help message and exit
  -o ouput_file    filename where results will be written
  -t num_results   number of results returned
  -m {1,2,3}       rating method (1 - avg, 2 - max, 3 - custom)
  -n NEIGHBORHOOD  number of neighbours selected when generating candidates
  -d {1,2}         similarity measure (1 - euclidean, 2 - cosine)
  -s {1,2,3}       normalization method applied to avg method (1 - none, 2 -
                   standard, 3 - softmax)
```

You can also calculate certain statistics from seed sets. These are also mentioned in more detail in the thesis. This code shows how can we work with seed sets in our library. **Warning:** Importing embeddings file will result in loading of word embeddings vector dataset. This could take up to several minutes depending on the size of dataset and the hardware.

```python
import model
output = model.PairSet.create_from_file(input_file)
set_recall = seed_set.seed_recall()
```

## Results

Result files have similar format to input files. There is however correctness flag at the start of the line:
```
?   Bogotá  Colombia
?   Managua Colombia
?   Havana  Colombia
```
When the file is generated the flag is set to *?*. Human evaulator can change these flags to:

* **t** - if given pair is correct
* **p** - if it is partially correct
* **f** - if it is incorrect
 
