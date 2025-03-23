# rag
> This project is my first rag based on maths lessons for students

## 1 Project requirements

### Fill your .env with the followings variables 

- OPENAI_API_KEY

### Pyenv and `Python 3.11.6`

- Install the right version of `Python` with `pyenv`:
  ```bash
  pyenv install 3.11.6
  ```

### Poetry

- Install [Poetry](https://python-poetry.org) to manage your dependencies and tooling configs:
  ```bash
  curl -sSL https://install.python-poetry.org | python - --version 1.7.0
  ```


## 2 Installation

### Python virtual environment and dependencies

```bash
make install
```

### Install git hooks (running before commit and push commands)

```bash
poetry run pre-commit install
```

## Initiate project 

### Create docker image for qdrant collection 

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 3 Launch rag

1. Create qdrant vectorial collection named `math_report` and fill it with pdf embeded informations 
2. Launch rag with open ai client 

```bash
python main.py
```

### Clik on web UI for qdrant collection visualisation

Link : http://localhost:6333/dashboard