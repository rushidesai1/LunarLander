# Pre-requisites

1. Please install dependencies from the ```requirements.txt``` file and then use below command


# To run Training:

Run below command

```bash
python lunar_lander/main.py
```


# To test the best config:

```bash
python lunar_lander/test.py
```

# To see previous experiments 

1. Install mlflow

2. Take the absolute path to ```mlruns``` dir 

```bash
 mlflow ui --backend-store-uri file:///<absolute_path_to_mlruns>
```

Once server is up go to browser to ```http://127.0.0.1:5000```

