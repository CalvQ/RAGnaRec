Link to google doc - https://docs.google.com/document/d/18o4e1PhFWSVfNhSOhTrFjIlDVLvDl4sJ80oHYUnDW7Y/edit?tab=t.0

---

# Clustering Installation Instructions

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

## Installing GuidedLDA

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

## Setting up Spacy

We also need a specific english package from `spaCy`\
Run this command **inside your conda env**.

```
python -m spacy download en_core_web_trf
```

## Downloading Data

Running `python preprocess_data.py` will download and store cleaned data in `.pkl` files.
