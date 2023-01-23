# HMER Project

## Setup

1. Install specified dependences using the following command : 

```python
pip install -r requirements.txt
```

2. In the src/utils/ directory, clone the repository ThomasLech/CROHME_extractor, that can be found here : https://github.com/ThomasLech/CROHME_extractor

3. Extract CROHME_full_v2.zip (found in src/utils/CROHME_extractor/data) before running

4. Go to the directory src/utils/CROHME_extractor. There, extract the datasets of isolated symbols by using the following command : 

```python 
python .\extract.py -b 50 -d 2011 -c all -t 4
```


