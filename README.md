# RAGnaRec

## Installation

Start by updating conda environment, replacing the environment name you want in `myenv`\
Leave empty to rename to default name: `ragnarec`

#### Update existing environment:

```
conda env update {--name myenv} --file environment.yml --prune
```

#### Create new environment:

```
conda env create -f environment.yml
```

#### Setting up spaCy

We also need a specific english package from `spaCy`\
Run this command **inside your conda env**.

```
python -m spacy download en_core_web_trf
```

## Usage

The fully-functional pipeline can be found in [E2E/ragnarec.ipynb](E2E/ragnarec.ipynb)

Running each cell will load the pipeline, and inputting a review along with a HuggingFace login token will generate a recommendation from RAGnaRec.

An example input/output can be found here:
```
Caliban Book Shop is so good - they seem to have a bit of everything packed into this cute place. The atmosphere is cozy and welcoming, and the staff have always been more than happy to help! love everything i've gotten from there :) It's close to CMU and Pitt, though, so watch out for packs of undergrads.
```

```
Based on your review, it seems like you're looking for places with a cozy atmosphere and friendly staff, similar to Caliban Book Shop. 

Here are a few recommendations:

The Book Nook: This charming bookstore is known for its welcoming atmosphere and knowledgeable staff. They have a great selection of books and host regular events. 

The Paper Trail: This independent bookstore is a local favorite, with a cozy atmosphere and a curated selection of books. They also offer a variety of literary events and workshops. 

The Reading Room: This cozy cafe is a great place to relax and enjoy a good book. They have a wide selection of coffee, tea, and pastries, and the staff is always friendly and helpful. 

Let me know if you'd like more recommendations!
```

---


## Old Experimentation Steps
### Installing GuidedLDA

Next we need to install `GuidedLDA`\
Run these commands **inside your conda env**.\
Feel free to ignore the warnings/errors (depreciation issues)

```
git clone https://github.com/vi3k6i5/GuidedLDA
cd GuidedLDA
sh build_dist.sh
python setup.py sdist
pip install -e .
```

## Downloading Data

Running `python preprocess_data.py` will download and store cleaned data in `.pkl` files.

---
Link to [Project Proposal](https://docs.google.com/document/d/18o4e1PhFWSVfNhSOhTrFjIlDVLvDl4sJ80oHYUnDW7Y/edit?tab=t.0)
