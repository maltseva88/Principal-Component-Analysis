# Clustering Crypto


```python
# Initial imports
import requests
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
```

### Fetching Cryptocurrency Data


```python
# Use the following endpoint to fetch json data
url = "https://min-api.cryptocompare.com/data/all/coinlist"
```


```python
## getting the data from the cryptowebsite
response = requests.get(url).json()
```


```python
import json
```


```python
crypto_df1 = pd.DataFrame(response["Data"]).T
crypto_df1 = crypto_df1[["Name", "CoinName", "Algorithm", "IsTrading", "ProofType", "TotalCoinsMined", "TotalCoinSupply"]]
crypto_df1.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>IsTrading</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42</td>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>300 token</td>
      <td>N/A</td>
      <td>True</td>
      <td>N/A</td>
      <td>300</td>
      <td>300</td>
    </tr>
    <tr>
      <th>365</th>
      <td>365</td>
      <td>365Coin</td>
      <td>X11</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>2300000000</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404</td>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>1.41363e+06</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>433</th>
      <td>433</td>
      <td>433 Token</td>
      <td>N/A</td>
      <td>False</td>
      <td>N/A</td>
      <td>1.12518e+08</td>
      <td>1000000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Alternatively, use the provided csv file:
# file_path = Path("Resources/crypto_data.csv")
file_path = Path("Resources/crypto_data.csv")
# Create a DataFrame
crypto_df2 = pd.read_csv(file_path)
crypto_df2.set_index("Unnamed: 0", inplace=True)
crypto_df2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>IsTrading</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>365</th>
      <td>365Coin</td>
      <td>X11</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>2300000000</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>611</th>
      <td>SixEleven</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>611000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>808</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Data Preprocessing


```python
# Keep only necessary columns:
# 'CoinName','Algorithm','IsTrading','ProofType','TotalCoinsMined','TotalCoinSupply'
crypto_df2 = crypto_df.drop(columns=["Unnamed: 0"])
crypto_df2.head()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-144-d09966672a40> in <module>
          1 # Keep only necessary columns:
          2 # 'CoinName','Algorithm','IsTrading','ProofType','TotalCoinsMined','TotalCoinSupply'
    ----> 3 crypto_df2 = crypto_df.drop(columns=["Unnamed: 0"])
          4 crypto_df2.head()


    ~/.local/lib/python3.7/site-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3995             level=level,
       3996             inplace=inplace,
    -> 3997             errors=errors,
       3998         )
       3999 


    ~/.local/lib/python3.7/site-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3934         for axis, labels in axes.items():
       3935             if labels is not None:
    -> 3936                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       3937 
       3938         if inplace:


    ~/.local/lib/python3.7/site-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors)
       3968                 new_axis = axis.drop(labels, level=level, errors=errors)
       3969             else:
    -> 3970                 new_axis = axis.drop(labels, errors=errors)
       3971             result = self.reindex(**{axis_name: new_axis})
       3972 


    ~/.local/lib/python3.7/site-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
       5015         if mask.any():
       5016             if errors != "ignore":
    -> 5017                 raise KeyError(f"{labels[mask]} not found in axis")
       5018             indexer = indexer[~mask]
       5019         return self.delete(indexer)


    KeyError: "['Unnamed: 0'] not found in axis"



```python
crypto_df2.dtypes
```


```python
# Keep only cryptocurrencies that are trading
crypto_df2 = crypto_df2.loc[crypto_df2["IsTrading"] == True, :]
crypto_df2.sample(5)
```


```python
# Keep only cryptocurrencies with a working algorithm
crypto_df2 = crypto_df2.loc[crypto_df2["Algorithm"] != "N/A", :]
crypto_df2.sample(5)
```


```python
# Remove the "IsTrading" column
crypto_df2 = crypto_df2.drop(columns=["IsTrading"])
crypto_df2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>365</th>
      <td>365Coin</td>
      <td>X11</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>2300000000</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>611</th>
      <td>SixEleven</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>611000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>808</td>
      <td>SHA-256</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove rows with at least 1 null value
crypto_df2= crypto_df2.dropna()
crypto_df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>808</td>
      <td>SHA-256</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
crypto_df2.isnull().sum()
```




    CoinName           0
    Algorithm          0
    ProofType          0
    TotalCoinsMined    0
    TotalCoinSupply    0
    dtype: int64




```python
# Remove rows with cryptocurrencies having no coins mined
crypto_df2 = crypto_df2.drop(crypto_df2[crypto_df2['TotalCoinsMined'] == 0].index)
crypto_df2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop rows where there are 'N/A' text values
crypto_df2 = crypto_df2[crypto_df2[:]!= "N/A"]
crypto_df2.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DAXX</th>
      <td>DaxxCoin</td>
      <td>Ethash</td>
      <td>PoW</td>
      <td>5.208918e+08</td>
      <td>10000000000</td>
    </tr>
    <tr>
      <th>GLC</th>
      <td>GlobalCoin</td>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>6.567272e+07</td>
      <td>70000000</td>
    </tr>
    <tr>
      <th>XJO</th>
      <td>JouleCoin</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>3.919574e+07</td>
      <td>45000000</td>
    </tr>
    <tr>
      <th>DFT</th>
      <td>Draftcoin</td>
      <td>Scrypt</td>
      <td>PoS</td>
      <td>1.866330e+07</td>
      <td>17405891.19707116</td>
    </tr>
    <tr>
      <th>PXC</th>
      <td>PhoenixCoin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>7.395927e+07</td>
      <td>98000000</td>
    </tr>
    <tr>
      <th>PRX</th>
      <td>Printerium</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.182173e+07</td>
      <td>20000000</td>
    </tr>
    <tr>
      <th>FIND</th>
      <td>FindCoin</td>
      <td>X13</td>
      <td>PoS</td>
      <td>1.452485e+07</td>
      <td>14524851.4827</td>
    </tr>
    <tr>
      <th>WOMEN</th>
      <td>WomenCoin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.845947e+10</td>
      <td>25000000000</td>
    </tr>
    <tr>
      <th>SPD</th>
      <td>Stipend</td>
      <td>C11</td>
      <td>PoW/PoS</td>
      <td>1.125134e+07</td>
      <td>19340594</td>
    </tr>
    <tr>
      <th>ROYAL</th>
      <td>RoyalCoin</td>
      <td>X13</td>
      <td>PoS</td>
      <td>2.500124e+06</td>
      <td>2500124</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store the 'CoinName'column in its own DataFrame prior to dropping it from crypto_df
coin_name = pd.DataFrame(crypto_df2['CoinName'], index =crypto_df2.index)
coin_name.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm
crypto_df2 = crypto_df2.drop(columns=['CoinName'])
```


```python
# Create dummy variables for text features
crypto_df = pd.get_dummies(crypto_df2, columns=["Algorithm", "ProofType"])
crypto_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>Algorithm_1GB AES Pattern Search</th>
      <th>Algorithm_536</th>
      <th>Algorithm_Argon2d</th>
      <th>Algorithm_BLAKE256</th>
      <th>Algorithm_Blake</th>
      <th>Algorithm_Blake2S</th>
      <th>Algorithm_Blake2b</th>
      <th>Algorithm_C11</th>
      <th>...</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoW</th>
      <th>ProofType_PoW/nPoS</th>
      <th>ProofType_Pos</th>
      <th>ProofType_Proof of Authority</th>
      <th>ProofType_Proof of Trust</th>
      <th>ProofType_TPoS</th>
      <th>ProofType_Zero-Knowledge Proof</th>
      <th>ProofType_dPoW/PoW</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>




```python
# Standardize data
crypto_df_scaled = StandardScaler().fit_transform(crypto_df)
print(crypto_df_scaled[0:5])
```

    [[-0.11416167 -0.15072664 -0.04163054 -0.04163054 -0.04163054 -0.05892557
      -0.09341308 -0.04163054 -0.05892557 -0.05892557 -0.04163054 -0.04163054
      -0.18932061 -0.05892557 -0.09341308 -0.04163054 -0.11072125 -0.07223151
      -0.04163054 -0.04163054 -0.15168668 -0.04163054 -0.13268622 -0.04163054
      -0.04163054 -0.08347839 -0.05892557 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.08347839 -0.09341308 -0.10241831 -0.04163054
      -0.12576654 -0.13268622 -0.15168668 -0.04163054 -0.08347839 -0.04163054
      -0.04163054 -0.07223151 -0.17407766 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.16872982 -0.30772873 -0.04163054 -0.09341308 -0.09341308
      -0.05892557 -0.04163054  1.40146444 -0.04163054 -0.04163054 -0.04163054
      -0.08347839 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.04163054 -0.39496835 -0.04163054 -0.17407766
      -0.04163054 -0.08347839 -0.08347839 -0.10241831 -0.04163054 -0.04163054
      -0.12576654 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.4434945  -0.04163054 -0.05892557 -0.04163054 -0.04163054
      -0.88852332 -0.04163054 -0.04163054  1.41238049 -0.04163054 -0.04163054
      -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.04163054]
     [-0.09006124 -0.142553   -0.04163054 -0.04163054 -0.04163054 -0.05892557
      -0.09341308 -0.04163054 -0.05892557 -0.05892557 -0.04163054 -0.04163054
      -0.18932061 -0.05892557 -0.09341308 -0.04163054 -0.11072125 -0.07223151
      -0.04163054 -0.04163054 -0.15168668 -0.04163054 -0.13268622 -0.04163054
      -0.04163054 -0.08347839 -0.05892557 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.08347839 -0.09341308 -0.10241831 -0.04163054
      -0.12576654 -0.13268622 -0.15168668 -0.04163054 -0.08347839 -0.04163054
      -0.04163054 -0.07223151 -0.17407766 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.16872982 -0.30772873 -0.04163054 -0.09341308 -0.09341308
      -0.05892557 -0.04163054  1.40146444 -0.04163054 -0.04163054 -0.04163054
      -0.08347839 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.04163054 -0.39496835 -0.04163054 -0.17407766
      -0.04163054 -0.08347839 -0.08347839 -0.10241831 -0.04163054 -0.04163054
      -0.12576654 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.4434945  -0.04163054 -0.05892557 -0.04163054 -0.04163054
      -0.88852332 -0.04163054 -0.04163054  1.41238049 -0.04163054 -0.04163054
      -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.04163054]
     [ 0.55458069  4.67601177 -0.04163054 -0.04163054 -0.04163054 -0.05892557
      -0.09341308 -0.04163054 -0.05892557 -0.05892557 -0.04163054 -0.04163054
      -0.18932061 -0.05892557 -0.09341308 -0.04163054 -0.11072125 -0.07223151
      -0.04163054 -0.04163054 -0.15168668 -0.04163054 -0.13268622 -0.04163054
      -0.04163054 -0.08347839 -0.05892557 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.08347839 -0.09341308 -0.10241831 -0.04163054
      -0.12576654 -0.13268622 -0.15168668 -0.04163054 -0.08347839 -0.04163054
      -0.04163054 -0.07223151 -0.17407766 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.16872982 -0.30772873 -0.04163054 -0.09341308 -0.09341308
      -0.05892557 -0.04163054 -0.71353933 -0.04163054 -0.04163054 -0.04163054
      -0.08347839 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.04163054 -0.39496835 -0.04163054  5.74456265
      -0.04163054 -0.08347839 -0.08347839 -0.10241831 -0.04163054 -0.04163054
      -0.12576654 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.4434945  -0.04163054 -0.05892557 -0.04163054 -0.04163054
      -0.88852332 -0.04163054 -0.04163054  1.41238049 -0.04163054 -0.04163054
      -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.04163054]
     [-0.11375222 -0.150404   -0.04163054 -0.04163054 -0.04163054 -0.05892557
      -0.09341308 -0.04163054 -0.05892557 -0.05892557 -0.04163054 -0.04163054
      -0.18932061 -0.05892557 -0.09341308 -0.04163054 -0.11072125 -0.07223151
      -0.04163054 -0.04163054 -0.15168668 -0.04163054 -0.13268622 -0.04163054
      -0.04163054 -0.08347839 -0.05892557 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.08347839 -0.09341308 -0.10241831 -0.04163054
      -0.12576654 -0.13268622 -0.15168668 -0.04163054 -0.08347839 -0.04163054
      -0.04163054 -0.07223151 -0.17407766 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.16872982  3.24961536 -0.04163054 -0.09341308 -0.09341308
      -0.05892557 -0.04163054 -0.71353933 -0.04163054 -0.04163054 -0.04163054
      -0.08347839 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.04163054 -0.39496835 -0.04163054 -0.17407766
      -0.04163054 -0.08347839 -0.08347839 -0.10241831 -0.04163054 -0.04163054
      -0.12576654 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.4434945  -0.04163054 -0.05892557 -0.04163054 -0.04163054
       1.12546287 -0.04163054 -0.04163054 -0.70802451 -0.04163054 -0.04163054
      -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.04163054]
     [-0.11170216 -0.15072664 -0.04163054 -0.04163054 -0.04163054 -0.05892557
      -0.09341308 -0.04163054 -0.05892557 -0.05892557 -0.04163054 -0.04163054
      -0.18932061 -0.05892557 -0.09341308 -0.04163054 -0.11072125 -0.07223151
      -0.04163054 -0.04163054 -0.15168668 -0.04163054  7.53657747 -0.04163054
      -0.04163054 -0.08347839 -0.05892557 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.08347839 -0.09341308 -0.10241831 -0.04163054
      -0.12576654 -0.13268622 -0.15168668 -0.04163054 -0.08347839 -0.04163054
      -0.04163054 -0.07223151 -0.17407766 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.16872982 -0.30772873 -0.04163054 -0.09341308 -0.09341308
      -0.05892557 -0.04163054 -0.71353933 -0.04163054 -0.04163054 -0.04163054
      -0.08347839 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.05892557 -0.04163054 -0.04163054 -0.39496835 -0.04163054 -0.17407766
      -0.04163054 -0.08347839 -0.08347839 -0.10241831 -0.04163054 -0.04163054
      -0.12576654 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.07223151 -0.4434945  -0.04163054 -0.05892557 -0.04163054 -0.04163054
       1.12546287 -0.04163054 -0.04163054 -0.70802451 -0.04163054 -0.04163054
      -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054 -0.04163054
      -0.04163054]]


### Reducing Dimensions Using PCA


```python
# Use PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)

# Get two principal components for the crypto data.
crypto_pca = pca.fit_transform(crypto_df_scaled)
```


```python
# Create a DataFrame with the principal components data (crypto_df.index)
pcs_df = pd.DataFrame(index =crypto_df2.index, 
    data=crypto_pca, columns=["PC 1", "PC 2", "PC 3"]
)
pcs_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>-0.333455</td>
      <td>1.147143</td>
      <td>-0.657732</td>
    </tr>
    <tr>
      <th>404</th>
      <td>-0.316105</td>
      <td>1.147435</td>
      <td>-0.658324</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.426675</td>
      <td>1.694263</td>
      <td>-0.686350</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>-0.140281</td>
      <td>-1.348490</td>
      <td>0.231902</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>-0.134493</td>
      <td>-2.055107</td>
      <td>0.367160</td>
    </tr>
  </tbody>
</table>
</div>



### Clustering Crytocurrencies Using K-Means

#### Find the Best Value for `k` Using the Elbow Curve


```python
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)

# Create the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")
```




<div id='1102'>





  <div class="bk-root" id="ab5915d0-d460-45f9-b132-897aa3bd8cb3" data-root-id="1102"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"8f791508-6a26-47b4-b8a1-d0433c9c5ffb":{"roots":{"references":[{"attributes":{"line_color":"#1f77b3","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1140","type":"Line"},{"attributes":{},"id":"1126","type":"WheelZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1106"},{"id":"1124"},{"id":"1125"},{"id":"1126"},{"id":"1127"},{"id":"1128"}]},"id":"1130","type":"Toolbar"},{"attributes":{},"id":"1138","type":"Selection"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"1104","type":"Range1d"},{"attributes":{},"id":"1150","type":"BasicTickFormatter"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"1145","type":"FixedTicker"},{"attributes":{},"id":"1157","type":"UnionRenderers"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01703","sizing_mode":"stretch_width"},"id":"1103","type":"Spacer"},{"attributes":{"align":null,"below":[{"id":"1116"}],"center":[{"id":"1119"},{"id":"1123"}],"left":[{"id":"1120"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1143"}],"sizing_mode":"fixed","title":{"id":"1108"},"toolbar":{"id":"1130"},"x_range":{"id":"1104"},"x_scale":{"id":"1112"},"y_range":{"id":"1105"},"y_scale":{"id":"1114"}},"id":"1107","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1114","type":"LinearScale"},{"attributes":{"callback":null,"renderers":[{"id":"1143"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"1106","type":"HoverTool"},{"attributes":{},"id":"1125","type":"PanTool"},{"attributes":{"axis_label":"inertia","bounds":"auto","formatter":{"id":"1150"},"major_label_orientation":"horizontal","ticker":{"id":"1121"}},"id":"1120","type":"LinearAxis"},{"attributes":{"end":4323.503591484156,"reset_end":4323.503591484156,"reset_start":-220.9434922267969,"start":-220.9434922267969,"tags":[[["inertia","inertia",null]]]},"id":"1105","type":"Range1d"},{"attributes":{},"id":"1121","type":"BasicTicker"},{"attributes":{},"id":"1124","type":"SaveTool"},{"attributes":{},"id":"1128","type":"ResetTool"},{"attributes":{},"id":"1146","type":"BasicTickFormatter"},{"attributes":{},"id":"1112","type":"LinearScale"},{"attributes":{"axis":{"id":"1116"},"grid_line_color":null,"ticker":null},"id":"1119","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b3","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1141","type":"Line"},{"attributes":{"axis_label":"k","bounds":"auto","formatter":{"id":"1146"},"major_label_orientation":"horizontal","ticker":{"id":"1145"}},"id":"1116","type":"LinearAxis"},{"attributes":{"data_source":{"id":"1137"},"glyph":{"id":"1140"},"hover_glyph":null,"muted_glyph":{"id":"1142"},"nonselection_glyph":{"id":"1141"},"selection_glyph":null,"view":{"id":"1144"}},"id":"1143","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"1120"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1123","type":"Grid"},{"attributes":{"source":{"id":"1137"}},"id":"1144","type":"CDSView"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1129","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.2,"line_color":"#1f77b3","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1142","type":"Line"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01704","sizing_mode":"stretch_width"},"id":"1166","type":"Spacer"},{"attributes":{"overlay":{"id":"1129"}},"id":"1127","type":"BoxZoomTool"},{"attributes":{"data":{"inertia":{"__ndarray__":"dzYQbpnRrkDHg5dM2OykQLDHKBxJkZlAsWsDsyqVg0DWPuvHqwp7QGU6MbjeBXVAVPpxaKgycEBb7oLlMpJrQDfy7Q+rxmdAYbRDdFW4Y0A=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"1138"},"selection_policy":{"id":"1157"}},"id":"1137","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"1103"},{"id":"1107"},{"id":"1166"}],"margin":[0,0,0,0],"name":"Row01699","tags":["embedded"]},"id":"1102","type":"Row"},{"attributes":{"text":"Elbow Curve","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1108","type":"Title"}],"root_ids":["1102"]},"title":"Bokeh Application","version":"2.1.1"}};
  var render_items = [{"docid":"8f791508-6a26-47b4-b8a1-d0433c9c5ffb","root_ids":["1102"],"roots":{"1102":"ab5915d0-d460-45f9-b132-897aa3bd8cb3"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>



Running K-Means with `k=<your best value for k here>`


```python
# Initialize the K-Means model

model = KMeans(n_clusters=4, random_state=5)

# Fit the model
model.fit(pcs_df)

# Predict clusters
predictions = model.predict(pcs_df)

# Create a new DataFrame including predicted clusters and cryptocurrencies features
clustered_df = pd.concat([crypto_df2, pcs_df, coin_name], axis = 1)
clustered_df["Class"] = model.labels_
clustered_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
      <th>CoinName</th>
      <th>Class</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>-0.333455</td>
      <td>1.147143</td>
      <td>-0.657732</td>
      <td>42 Coin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>-0.316105</td>
      <td>1.147435</td>
      <td>-0.658324</td>
      <td>404Coin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>2.426675</td>
      <td>1.694263</td>
      <td>-0.686350</td>
      <td>EliteCoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>-0.140281</td>
      <td>-1.348490</td>
      <td>0.231902</td>
      <td>Bitcoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>-0.134493</td>
      <td>-2.055107</td>
      <td>0.367160</td>
      <td>Ethereum</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Visualizing Results

#### 3D-Scatter with Clusters


```python
# Create a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_df,
    x="PC 1",
    y="PC 2",
    z= "PC 3",
    hover_name="CoinName",
    hover_data=["Algorithm"],
    color="Class",
    symbol="Class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
```


<div>


            <div id="01225515-950e-4ee9-a340-ac7008e2b7a1" class="plotly-graph-div" style="height:525px; width:800px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("01225515-950e-4ee9-a340-ac7008e2b7a1")) {
                    Plotly.newPlot(
                        '01225515-950e-4ee9-a340-ac7008e2b7a1',
                        [{"customdata": [["Scrypt"], ["Scrypt"], ["X13"], ["X11"], ["SHA-512"], ["SHA-256"], ["SHA-256"], ["X15"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Groestl"], ["PoS"], ["Scrypt"], ["Scrypt"], ["X11"], ["X11"], ["SHA3"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["X13"], ["X13"], ["NeoScrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X11"], ["X11"], ["Multiple"], ["PHI1612"], ["X11"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X11"], ["Multiple"], ["X13"], ["Scrypt"], ["Shabal256"], ["Counterparty"], ["SHA-256"], ["Groestl"], ["Scrypt"], ["X13"], ["Scrypt"], ["Scrypt"], ["X13"], ["X11"], ["Scrypt"], ["X11"], ["SHA3"], ["QUAIT"], ["X11"], ["Scrypt"], ["X13"], ["SHA-256"], ["X15"], ["BLAKE256"], ["SHA-256"], ["X11"], ["SHA-256"], ["NIST5"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["PoS"], ["X11"], ["SHA-256"], ["SHA-256"], ["NIST5"], ["X11"], ["POS 3.0"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X13"], ["X11"], ["X11"], ["Scrypt"], ["SHA-256"], ["X11"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["SHA-256D"], ["PoS"], ["Scrypt"], ["X11"], ["PoS"], ["X13"], ["X14"], ["PoS"], ["SHA-256D"], ["DPoS"], ["X11"], ["X13"], ["X11"], ["PoS"], ["Scrypt"], ["Scrypt"], ["PoS"], ["X11"], ["SHA-256"], ["Scrypt"], ["X11"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Quark"], ["QuBit"], ["Scrypt"], ["SHA-256"], ["X11"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X13"], ["Scrypt"], ["Scrypt"], ["X11"], ["Blake2S"], ["X11"], ["PoS"], ["X11"], ["PoS"], ["X11"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["SHA-256"], ["X11"], ["Scrypt"], ["PoS"], ["Scrypt"], ["X15"], ["SHA-256"], ["POS 3.0"], ["536"], ["NIST5"], ["NIST5"], ["Skein"], ["X13"], ["Scrypt"], ["X13"], ["SkunkHash v2 Raptor"], ["Skein"], ["X11"], ["Scrypt"], ["PoS"], ["Scrypt"], ["Scrypt"], ["SHA-512"], ["Ouroboros"], ["X11"], ["NeoScrypt"], ["Scrypt"], ["Lyra2REv2"], ["Scrypt"], ["SHA-256"], ["NIST5"], ["PHI1612"], ["Scrypt"], ["Quark"], ["POS 2.0"], ["Scrypt"], ["SHA-256"], ["Quark"], ["X11"], ["DPoS"], ["NIST5"], ["X13"], ["Scrypt"], ["NIST5"], ["Quark"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["Scrypt"], ["Quark"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["POS 3.0"], ["Scrypt"], ["Scrypt"], ["X11"], ["SHA-256"], ["X13"], ["Proof-of-BibleHash"], ["Skein"], ["X11"], ["Skein"], ["C11"], ["X11"], ["XEVAN"], ["Scrypt"], ["VBFT"], ["PHI1612"], ["NIST5"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Green Protocol"], ["PoS"], ["Scrypt"], ["Semux BFT consensus"], ["Quark"], ["PoS"], ["X16R"], ["Scrypt"], ["XEVAN"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X11"], ["SHA-256D"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["SHA-256D"], ["Scrypt"], ["X15"], ["Scrypt"], ["Quark"], ["Scrypt"], ["SHA-256"], ["DPoS"], ["Scrypt"], ["X16R"], ["Quark"], ["Quark"], ["Scrypt"], ["Lyra2REv2"], ["Quark"], ["Scrypt"], ["X11"], ["X11"], ["Scrypt"], ["PoS"], ["Keccak"], ["X11"], ["Tribus"], ["Scrypt"], ["NeoScrypt"], ["SHA-512"], ["XEVAN"], ["SHA-512"], ["Quark"], ["XEVAN"], ["X11"], ["Quark"], ["Scrypt"], ["Quark"], ["Quark"], ["Scrypt"], ["X11"], ["Scrypt"], ["XEVAN"], ["SHA-256D"], ["X11"], ["X11"], ["DPoS"], ["Scrypt"], ["X11"], ["Scrypt"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["X11"], ["Scrypt"], ["SHA-256"], ["X11"], ["Scrypt"], ["Scrypt"], ["X11"], ["Scrypt"], ["PoS"], ["X11"], ["SHA-256"], ["DPoS"], ["Scrypt"], ["Scrypt"], ["NeoScrypt"], ["SHA3-256"], ["X13"], ["PHI2"], ["DPoS"], ["DPoS"], ["SHA-256"], ["Lyra2Z"], ["PoS"], ["Scrypt"], ["PoS"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Scrypt"]], "hovertemplate": "<b>%{hovertext}</b><br><br>Class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>", "hovertext": ["42 Coin", "404Coin", "EliteCoin", "Dash", "Bitshares", "BitcoinDark", "PayCoin", "KoboCoin", "Aurora Coin", "BlueCoin", "EnergyCoin", "BitBar", "CryptoBullion", "CasinoCoin", "Diamond", "Exclusive Coin", "FlutterCoin", "HoboNickels", "HyperStake", "IOCoin", "MaxCoin", "MintCoin", "MazaCoin", "Nautilus Coin", "NavCoin", "OpalCoin", "Orbitcoin", "PotCoin", "PhoenixCoin", "Reddcoin", "SuperCoin", "SyncCoin", "TeslaCoin", "TittieCoin", "TorCoin", "UnitaryStatus Dollar", "UltraCoin", "VeriCoin", "X11 Coin", "Crypti", "StealthCoin", "ZCC Coin", "BurstCoin", "StorjCoin", "Neutron", "FairCoin", "RubyCoin", "Kore", "Dnotes", "8BIT Coin", "Sativa Coin", "Ucoin", "Vtorrent", "IslaCoin", "Nexus", "Droidz", "Squall Coin", "Diggits", "Paycon", "Emercoin", "EverGreenCoin", "Decred", "EDRCoin", "Hitcoin", "DubaiCoin", "PWR Coin", "BillaryCoin", "GPU Coin", "EuropeCoin", "ZeitCoin", "SwingCoin", "SafeExchangeCoin", "Nebuchadnezzar", "Ratecoin", "Revenu", "Clockcoin", "VIP Tokens", "BitSend", "Let it Ride", "PutinCoin", "iBankCoin", "Frankywillcoin", "MudraCoin", "Lutetium Coin", "GoldBlocks", "CarterCoin", "BitTokens", "MustangCoin", "ZoneCoin", "RootCoin", "BitCurrency", "Swiscoin", "BuzzCoin", "Opair", "PesoBit", "Halloween Coin", "CoffeeCoin", "RoyalCoin", "GanjaCoin V2", "TeamUP", "LanaCoin", "ARK", "InsaneCoin", "EmberCoin", "XenixCoin", "FreeCoin", "PLNCoin", "AquariusCoin", "Creatio", "Eternity", "Eurocoin", "BitcoinFast", "Stakenet", "BitConnect Coin", "MoneyCoin", "Enigma", "Russiacoin", "PandaCoin", "GameUnits", "GAKHcoin", "Allsafe", "LiteCreed", "Klingon Empire Darsek", "Internet of People", "KushCoin", "Printerium", "Impeach", "Zilbercoin", "FirstCoin", "FindCoin", "OpenChat", "RenosCoin", "VirtacoinPlus", "TajCoin", "Impact", "Atmos", "HappyCoin", "MacronCoin", "Condensate", "Independent Money System", "ArgusCoin", "LomoCoin", "ProCurrency", "GoldReserve", "GrowthCoin", "Phreak", "Degas Coin", "HTML5 Coin", "Ultimate Secure Cash", "QTUM", "Espers", "Denarius", "Virta Unique Coin", "Bitcoin Planet", "BritCoin", "Linda", "DeepOnion", "Signatum", "Cream", "Monoeci", "Draftcoin", "Stakecoin", "CoinonatX", "Ethereum Dark", "Obsidian", "Cardano", "Regalcoin", "TrezarCoin", "TerraNovaCoin", "Rupee", "WomenCoin", "Theresa May Coin", "NamoCoin", "LUXCoin", "Xios", "Bitcloud 2.0", "KekCoin", "BlackholeCoin", "Infinity Economics", "Alqo", "Magnet", "Lamden Tau", "Electra", "Bitcoin Diamond", "Cash & Back Coin", "Bulwark", "Kalkulus", "GermanCoin", "LiteCoin Ultra", "PhantomX", "Accolade", "OmiseGO Classic", "Digiwage", "Trollcoin", "Litecoin Plus", "Monkey Project", "ECC", "TokenPay", "My Big Coin", "Unified Society USDEX", "BitSoar Coin", "Credence Coin", "Tokyo Coin", "BiblePay", "BashCoin", "DigiMoney", "Lizus Payment", "Stipend", "Pushi", "Ellerium", "Velox", "Ontology", "Seraph", "Bitspace", "Briacoin", "Ignition", "MedicCoin", "Bitcoin Green", "Deviant Coin", "Abjcoin", "Semux", "Carebit", "Zealium", "Proton", "iDealCash", "Bitcoin Incognito", "HollyWoodCoin", "Parlay", "Listerclassic Coin", "BetKings", "Cognitio", "Mercoin", "Swisscoin", "Reliance", "Xt3ch", "TheVig", "EmaratCoin", "Dekado", "Lynx", "Poseidon Quark", "MYCE", "BitcoinWSpectrum", "Muse", "GambleCoin", "Trivechain", "Dystem", "Giant", "Peony Coin", "Absolute Coin", "Vitae", "TPCash", "ARENON", "EUNO", "MMOCoin", "Ketan", "XDNA", "PAXEX", "Averopay", "ThunderStake", "SimpleBank", "Kcash", "Bettex coin", "TWIST", "DACH Coin", "BitMoney", "Junson Ming Chan Coin", "HerbCoin", "Oduwa", "Galilel", "Crypto Sports", "Credit", "Dash Platinum", "Nasdacoin", "Beetle Coin", "Titan Coin", "Award", "Insane Coin", "ALAX", "LiteDoge", "TruckCoin", "OrangeCoin", "BitstarCoin", "NeosCoin", "HyperCoin", "PinkCoin", "AudioCoin", "IncaKoin", "Piggy Coin", "Genstake", "XiaoMiCoin", "CapriCoin", " ClubCoin", "Radium", "Creditbit ", "OKCash", "Lisk", "HiCoin", "WhiteCoin", "FriendshipCoin", "Fiii", "Triangles Coin", "Gexan", "EOS", "Oxycoin", "TigerCash", "LAPO", "Particl", "ShardCoin", "Nxt", "ZEPHYR", "Gapcoin", "BitcoinPlus", "DivotyCoin"], "legendgroup": "3", "marker": {"color": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "coloraxis": "coloraxis", "symbol": "circle"}, "mode": "markers", "name": "3", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [-0.33345506089064486, -0.31610499701936073, 2.4266749990360355, -0.3792069060772611, -0.2393745519822435, -0.3044922887363438, -0.25345146309003636, -0.22426904317405216, -0.33310074207395357, -0.32536522186031314, -0.3318882992070176, -0.3334508391144508, -0.3334344712712869, 0.7406817655318442, -0.36041682426789423, -0.32480954302266063, -0.3184809954950164, -0.3314352424994147, -0.30713946171358514, -0.3790971661975837, -0.3935679592113795, -0.011378997006019359, -0.2663163929467346, -0.28215551870576894, -0.2577929605841749, -0.25844138042730946, -0.34341030715501397, -0.32752842778444285, -0.3317920735591113, 0.039773336221071806, -0.3278638032060018, -0.3794838078418637, -0.22485946755754016, -0.2710391247386445, -0.37939199024178777, -0.307686174668527, -0.33207939215772764, -0.31947053786482155, -0.37935485057462975, -0.22534426019097045, -0.25821301259456525, -0.324209890971595, 0.13605857698449, -0.321196425483747, -0.3036720154210982, -0.3598139041519591, -0.2821345178059025, -0.2585202608751703, -0.2765766005019869, -0.3334363740543019, -0.2584703791138474, -0.3283279170249708, -0.33316009636503474, -0.37946455860706113, -0.37854982030818646, -0.3896956038538372, -0.35962551095448986, -0.2804713692738673, -0.2579729251601366, -0.29676138214156106, -0.2268047581680198, -0.05504229003588943, -0.304461977424142, -0.044049440737367836, -0.30452805024023144, -0.20940844793788257, -0.3330313311075789, -0.2803381061367588, -0.3765246889945812, 0.9169588585648016, -0.3043204725132642, -0.01986811552876483, -0.3244800045898371, -0.3772186780377556, -0.3030159309070753, -0.30059395003530687, -0.3358817600529181, -0.3781501546370796, -0.3236030155473824, -0.30838115895822127, -0.33307109552063346, -0.33144581640894677, -0.20612402871200844, -0.3153086412636328, -0.3789176278508784, -0.3322430099921164, -0.30450847816054966, -0.37945337441150895, -0.3332676076402174, -0.33343036421939803, -0.28032138570307114, -0.2572788469618579, 0.054684016009427926, -0.32339458170597946, -0.33302826734696933, -0.3650725308221964, -0.32269403831417665, -0.20760966150068008, -0.38913440190000176, -0.32243932165250766, -0.2727602167988526, 3.900654156723828, -0.3790294689746749, 0.9213668360788821, -0.3794064073996283, -0.3238772309921631, -0.3329537923447785, -0.3331147111989492, -0.3244800045898371, -0.3789648943306678, -0.3043653396615903, -0.3329544122954828, -0.34992146399650204, -0.33310787924190044, -0.3285265682922639, -0.37943665304569635, -0.33228840887432626, 0.38735208778939734, -0.3333151521669144, -0.3333884393990387, -0.33187791347032813, -0.33636745820657765, -0.32946942539147434, -0.3044838982775644, -0.37934292669207464, -0.3331573335734686, -0.33328968588939567, -0.2820385480335481, -0.30807394308848374, -0.20736805522582008, -0.31336260847872266, -0.28176822823906794, -0.3785801451897387, -0.39260707632184255, -0.3772656346057559, -0.3226550880714009, -0.3784737288013514, -0.3168267759189166, -0.3742543344305321, -0.3332305642364247, -0.3332299160135044, -0.32608764912367977, 0.29966241315260045, -0.3789707724696679, -0.26400103123410257, -0.32451600865244745, -0.33241022470564363, 0.952970125035929, -0.25209168210288974, -0.32233166782823003, 0.6402704146394146, -0.337479977236765, -0.33592202850297664, -0.40136845054957565, -0.25814273813163174, 0.14975197487835068, -0.2582161973520856, -0.3089270589792594, -0.4008524792366079, -0.3792538223311396, -0.2821148789955302, -0.32437748779768455, -0.3328511100644351, -0.33337067343085275, -0.2997915585666444, 0.7092234263874431, -0.3791926213298003, -0.3382093988281563, -0.3333247247725389, -0.2807636224182832, 0.4675283461852474, -0.3027625999639233, -0.3216366808382721, -0.3590510705257323, -0.3332731111980164, -0.3812334717367365, -0.31980513022792484, -0.33313788633209107, -0.07286411448415497, -0.3818396571954466, -0.3779375176003089, 3.9057047826949147, 0.2447858332047461, -0.2547518761601718, -0.33049634889070645, -0.28625957324596085, -0.3317591435414978, 0.09712758935796083, -0.3322437138662192, -0.37854064656612746, -0.3330160457014485, -0.33230406836553056, -0.3308913462643881, -0.319305568751844, -0.3333937319150381, -0.3282986905515445, 0.21983071679566565, -0.32389658240978636, -0.297988529202334, -0.32875737561293317, -0.25235705579589224, -0.3044429840155259, -0.19883902840343232, -0.25042926974374025, -0.4015127172798104, -0.37925496871254627, -0.40164525649113114, -0.33687068403422116, -0.3792722045273815, -0.3249424796951848, -0.28124230255535004, -0.2830230726647294, -0.3593456061442689, -0.3370631041397982, -0.33342221243517745, -0.3334032060234146, -0.2753970158125654, -0.31563611748317283, -0.3239555618103417, -0.3331084376366782, 4.086437978034738, -0.3798453913800569, -0.32415196634207233, -0.30020628246070036, -0.27787523846094536, -0.3052419349193711, -0.28179643526009024, -0.28221161684452295, -0.3325422881524615, -0.32830478009466213, -0.3417260945774303, -0.3323964673085351, -0.12851203852532098, -0.3279606116757428, -0.28205958790487784, -0.3406304136216201, -0.33256174382405695, -0.17514810022480046, 2.3039913095705264, -0.3273029728497088, -0.330804905547909, -0.2522221435198754, 1.0935266452030281, -0.2822608657590179, -0.350490330891968, -0.33187748441388254, -0.38298340216540916, -0.15821287651396942, -0.3316640904921531, -0.330545171764874, -0.3260252168214533, -0.3278591975115989, -0.3787240181338011, -0.2791947568113464, -0.32321728630191665, -0.34720510763067075, -0.3277159279723683, -0.3485801692263105, -0.1874615112880348, -0.34329415981660927, -0.28534256209313724, -0.2739605677810189, -0.29676407139253574, -0.3316604166651114, 0.4881434012288144, -0.10940285084702789, -0.38224961202888386, -0.33311234076330054, -0.3827240181859096, -0.3319991865422173, 0.6096058869394735, -0.32835525907744745, -0.3325751475946488, -0.3191761946670096, -0.2935725143327728, -0.37619586921465425, -0.3789602973683851, 3.9184489144170542, 0.11921255626357671, -0.37639552068463683, -0.3319378194590283, -0.3327937117074125, -0.2534856920494987, -0.33333244301379494, -0.3702479114290731, -0.2436779648586843, -0.07487411255581251, -0.36583040452712173, -0.3325807616500242, -0.3253631256280445, -0.3753890813544821, -0.33095804201726103, -0.32476695440049297, -0.36744584572208766, -0.30294699954151216, 3.9010615513260984, -0.08148792742648273, -0.32803836242249945, -0.3430218966619402, -0.6193448473439803, -0.25863166663595144, -0.40783892779930825, 3.9113494281808405, 3.912645966088775, -0.23360374040327714, 0.6356764443634252, -0.3247001099489963, -0.27553834225634916, -0.31993913443525746, 2.5916832768968177, -0.33142470679181124, -0.28247161976940416, -0.3324453436722295], "y": [1.1471428732890672, 1.1474348943328592, 1.6942634994301433, 1.156391829844598, 0.9322011604454968, 0.7978348901758366, 0.5032461146965541, 1.8161802891429912, 1.1471448779947993, 1.1474453617923888, 1.1472014563558828, 1.147142695913046, 1.147142972664412, 0.7919875185182733, 0.8873925724143651, 1.3649247015637376, 0.7688423460933436, 1.1471379339611683, 0.8625976455874057, 1.156395933146131, 2.570970085847731, 0.8626820501536673, 0.7976549544901217, 0.8525465525295677, 1.8045390103042052, 1.804514765133334, 0.4581514594170924, 1.1470828575221659, 1.1471393431305323, 1.1610983164732298, 0.8618227383941663, 1.1563962269925996, 0.0013878906289261775, 1.0291354893119569, 1.1563929556038817, 1.1470335657476356, 1.1471272589093149, 0.5551939331484322, 1.1563973616508296, 0.001436815928673534, 1.8045233040717779, 1.1468180383634135, 1.3316573382158867, 1.4822177595469768, 0.7978347171083304, 0.887418053186558, 0.8525581868331459, 1.8045037694306836, 0.8524307425186907, 1.1471435720397383, 1.8045069756128465, 0.861791974187703, 1.1471404919315007, 1.156396947414364, 3.5587292282852543, 1.7198644757094343, 0.4430336886450476, 0.8525533216287393, 1.804498755090114, 0.7974681866550178, 1.816302525319152, 0.879469454652905, 0.7978360235509961, 1.1511361333214059, 0.7978412640214929, 1.420618511430304, 1.1471305551169633, 0.852477135967546, 1.1562493918774084, 0.8310119438938656, 0.7978292451772633, 0.78893064272086, 1.3649236129365472, 1.1564306343023028, 0.7977555022620283, 0.7976433807225608, 1.4158291803978247, 1.156352891864718, 1.3469220865366118, 1.1467393704410531, 1.1471275037268926, 1.147150949042262, 1.5098332459629367, 0.8618516571348376, 1.1563838716608033, 1.1471278462744205, 0.7978349553570583, 1.1563953540366445, 1.1471358014374882, 1.1471437967544436, 0.8526259819478567, 0.6139502528395786, 1.8753141259795112, 1.3649279899385611, 1.1471588316234136, 1.155929298156386, 1.364976981905979, 1.5099101245567363, 1.9440172985017508, 1.3648114995927076, 1.8714481151609892, 1.5390668020508596, 1.1563931002431653, 1.848059220122971, 1.1563965380128738, 1.3649260356635236, 1.1471357743895114, 1.147127437438527, 1.3649236129365472, 1.156375399086314, 0.7978409779943373, 1.147139465903997, 0.3269367077857514, 1.147137080209629, 1.146890873260351, 1.156394638222744, 1.1470899405375057, 0.8557890820434906, 1.147139387873167, 1.1471431410607231, 1.2641407272348517, 0.5241893683993625, 1.1469566395603805, 0.7978358744274249, 1.1563952233045725, 1.1471405952354856, 1.1471343500852516, 0.8525248965117876, 0.594231456113595, 1.5099110956444581, 1.1472236308515464, 0.852548799130981, 1.1563629643997608, 1.8360225634556984, 1.1564054102509636, 1.3649313071792524, 1.1563669434320918, 1.3649549774582483, 1.156256502450139, 1.1471370443014477, 1.1471321147994753, 1.1467478268056788, 0.4736468718958749, 1.1563885897477206, 0.851895173767516, 1.3649155614769144, 1.1471115360832274, 1.800086278223257, 0.5031711790425352, 1.3469250339517052, 1.9615265852888346, 1.4158230630088877, 1.4158075590359445, 1.9123124702952798, 1.804505816037775, 1.131684415855409, 1.8045105133282389, 1.354157767399467, 1.912331763061811, 1.156398452248426, 0.8525472501077802, 1.3648995524420688, 1.1471331016837885, 1.1471432124674512, 0.9322950839282002, 1.360586900486273, 1.156389011354934, 0.45808024625185084, 1.1471371971807978, 0.3274096280731711, 1.1603295081878868, 0.7978472644853168, 1.415617540373232, 1.3238502122505265, 1.1471355956530715, 1.5586748401034023, 1.4200703598134576, 1.1471448169691159, 0.5039721635568921, 1.5587474692654126, 1.1563574899974722, 1.5390042048484676, 1.4174856852329647, 1.8045119102376699, 1.1471126932501077, 1.1212227531484216, 1.2641418155674438, 1.1297167289740444, 1.1470875886002032, 1.156397967428786, 1.1471257624753082, 1.1471389736827882, 1.2641072112560627, 1.1470684689252921, 1.1471424843884708, 0.8617923964768892, 0.8545641850189498, 1.3469168091727768, 0.6492992658411633, 1.1471629639832026, 0.8619773434393007, 0.7978347221688219, 1.5097033270282465, 6.338471392006659, 1.9123258506273049, 1.156388021159081, 1.9123229064000673, 0.8969676716249051, 1.1563873766919504, 1.2046596476849438, 0.8525084029959278, 1.3467722780085303, 1.3238579738543275, 1.4158118294743531, 1.1471420899913443, 1.1471414596179645, 0.8524748485599469, 1.3085780806543117, 1.3648976269247608, 1.1471357182854152, 1.545587488833988, 1.5587267420397597, 1.3648956473119962, 0.6059530260951435, 1.1457866834059593, 0.4913173605133446, 0.8525533945447069, 0.8525351883327915, 1.1471032454646384, 0.8618007862449564, 1.8738973145681208, 1.1471154031012425, 1.147966600457568, 0.8617775462123376, 0.8525314855550844, 1.8738762600115466, 1.1471199515818895, 1.5216876929297354, 0.5627243903043583, 1.2638860077308065, 1.1470743350547046, 0.5032332338415356, 1.4228169248491767, 0.8525430023840225, 0.9005512924984528, 1.264136720142216, 1.558740058829296, 0.8458733395365389, 0.6219909117558107, 1.2641335655744228, 1.1467501612219544, 0.8617860318638431, 1.1563911109529417, 0.852493772036305, 1.364843428147017, 0.8758468374220372, 0.8617612153616169, 0.9461436964860721, 1.1405323349984176, 0.45814424919667607, 0.6298245048534595, 0.9100690046141031, 0.9323354590236211, 1.2641334376733269, 0.4011410505175378, 0.8699912470769049, 1.5587347417364485, 1.14714160704631, 1.5587404472919573, 1.264137285645699, 1.1322499273107942, 0.8617911529933896, 1.1471194503993432, 1.2045802258372005, 1.8723502488059285, 1.1562375480174725, 1.1563956866498457, 1.5391454614356492, 1.1406003504183213, 1.15651170230601, 1.1470655002176657, 1.1471312219775707, 0.5032391353939778, 1.1471474581420258, 1.1564063072583701, 1.1434592559650398, 0.8063079326173531, 1.1562362216747561, 1.147165506542173, 1.1471772311359638, 1.156409865541997, 1.1471289561376163, 1.3649202592984417, 0.4875572417944394, 0.7978370169580267, 1.5390586214320061, 0.853349445418177, 1.1471442531850145, 0.4581281663060504, 2.382710907075184, 1.8045075696483228, 1.8686227983257764, 1.5395505259939057, 1.539599005051806, 0.503326103034677, 0.17372416085436998, 1.3649230040074973, 0.8522013551631233, 1.3889598094281346, 0.6196561537161936, 1.1470511599664894, 0.8525449116766994, 1.1471135755572495], "z": [-0.6577320663441719, -0.6583235722778025, -0.6863498760829023, -0.5028580840827264, -0.045334516753243854, -0.349002866684744, 0.034226208174626016, -0.7028509176991151, -0.6577433596383866, -0.6580409039816919, -0.6577918790316166, -0.657732160931935, -0.6577327192022655, 0.05681642268877912, -0.4799980682316378, 0.09695101829231366, -0.3548495247352745, -0.6577931937234989, -0.12043338596411945, -0.5028622735115252, -0.4466127813064805, -0.284848765460147, -0.3501410183309292, -0.2745094482747012, -0.6257063252520491, -0.6256815711811586, -0.31134700168036267, -0.6579023872580232, -0.6577825012299399, -0.671980430736323, -0.11964221366442139, -0.5028504428361444, 0.14085076865425725, -0.09654668489638042, -0.5028526164308526, -0.6585027365169801, -0.6577712662040618, -0.5299723470469795, -0.502854633569402, 0.14085595842950813, -0.6256902893497661, -0.6579518332391778, 1.1018478476198041, -0.05801238569508588, -0.3490280553067544, -0.48002166861900414, -0.2745124048030159, -0.6256769617220475, -0.2746579954880315, -0.6577327797382818, -0.6256791323610711, -0.11962183212596579, -0.6577406633802885, -0.5028511778279597, 0.05140783947083686, -0.7594122024574875, 0.017464782759330572, -0.2745625794838913, -0.625692796093519, -0.34916775495546554, -0.7027972238097334, -0.4694527166631982, -0.3490040238491624, -0.512120142821058, -0.3490030329852727, -0.6007863789793692, -0.657742649235554, -0.2745515455529138, -0.5029122704355429, -0.30710449630880843, -0.34900702875992035, 0.08081009788228842, 0.09694110136857541, -0.5029269282986976, -0.34903249622758, -0.34908470177998074, -0.5959461489012037, -0.5028828449499687, 0.05471104757459204, -0.6584229341439237, -0.6577408204441685, -0.6577954535869218, -0.24247332846383415, -0.12003402237300656, -0.5028653986199784, -0.6577663516973434, -0.34900238181405635, -0.5028512052657851, -0.6577364258540372, -0.6577330091696921, -0.27458162292007804, -0.5604105085223955, -0.7511381640069666, 0.0969068558118582, -0.6577483596167681, -0.5032008428004767, 0.09687558385324666, -0.2424429153290239, -0.8265054212276263, 0.096900618910021, -0.7403015645302938, -0.09743540406463619, -0.5028637925070167, -0.6706087203336465, -0.502852884634868, 0.09692208518776212, -0.6577460701459537, -0.6577394661178343, 0.09694110136857541, -0.502862262408758, -0.34900797944492534, -0.6577467842773367, -0.20208580290631184, -0.6577415914076109, -0.6578335637509365, -0.5028515772672251, -0.6577574270111102, -0.29574049269096603, -0.6577356762077202, -0.657734168105652, -0.12503175348488546, -0.3438693065046312, -0.6578176336168235, -0.34900332017547797, -0.5028545755126579, -0.657740768852598, -0.6577354586968103, -0.27450874382879464, -0.3869028168161711, -0.24245053747542486, -0.6583659390112963, -0.27452180347513877, -0.5028716235231496, -0.7482926667152184, -0.5029204745025287, 0.09688345789307537, -0.5028756860743652, 0.09669953889563708, -0.5029834949828469, -0.6577378117731691, -0.6577368526267794, -0.6578801479585621, 0.023097120842675924, -0.5028647015378802, -0.2749383153543261, 0.09694380762584451, -0.6577579704600567, -0.7358539047598635, 0.03419927902074419, 0.05467136886087824, -0.7881053206174106, -0.5958957895060614, -0.5959406163138014, -0.6988689535239149, -0.6256889768477504, -0.6695201421136975, -0.6256876509712533, 0.06335973200180178, -0.6988886512407476, -0.5028579567479308, -0.27451083647588564, 0.09694272783210789, -0.657748696732308, -0.6577347285828603, -0.043495378607794524, 0.015337539579069895, -0.5028579635339121, -0.3114927829339532, -0.6577349467478821, 0.027986136162728874, -0.6849809865327326, -0.34905851149152023, -0.5963421428580078, -0.4786639314248211, -0.6577362157502575, -0.508301792597356, -0.053652541784078515, -0.6577422053502544, 0.028529032059973227, -0.5082975780240845, -0.5028902966966037, -0.0975782754803662, -0.6141303969590411, -0.6257944545718142, -0.6578170510205072, -0.2126699456764761, -0.12503562176153776, -0.667511153981187, -0.6577583342275188, -0.5028797902545743, -0.6577421673579353, -0.6577666842662657, -0.12505543311908782, -0.6581523782706657, -0.6577338749376763, -0.11962281469894262, -0.29034600955825285, 0.05472112277936209, -0.5547916281945228, -0.6578805081646384, -0.12199471561729124, -0.3490043494114455, -0.24267153459165475, 3.62705889893062, -0.6988671749444183, -0.5028558497140044, -0.6988625146584538, -0.5621046766368613, -0.5028551917196713, -0.36541937235379557, -0.27452995210143777, -0.06362573421874397, -0.4786564161814128, -0.595906376984349, -0.6577329208422067, -0.6577333800775433, -0.2747030273039647, 0.04515182268628535, 0.09693013625699881, -0.6577413037370453, -0.19136801774116996, -0.5083547839422615, 0.09693656878171135, 0.0027947706010841202, -0.6591717569366471, 0.15489668182906394, -0.2745218488468726, -0.2745054661713509, -0.6577522629237101, -0.1196242937923286, -0.7386673487929595, -0.6577591615455693, -0.6641975677937914, -0.11963026095300978, -0.27450940555393116, -0.7386968586608982, -0.6577549827769266, -0.3195819101625364, -0.606029383088295, -0.12512183921253334, -0.657799944498139, 0.03419096555633978, -0.028959665961629983, -0.27450550378886973, -0.38045945983205076, -0.12503097080466774, -0.5082609366031222, -0.27699521464086113, -0.35524576662996943, -0.1250713122070012, -0.6578825313736464, -0.11963506477587062, -0.5028727898630323, -0.2745900071680493, 0.09691819940768408, -0.46615756823606613, -0.11963454128363721, -0.5071398230957919, -0.6609083353446784, -0.31134914108664796, -0.15522501403593966, 0.017811885007431593, -0.04359649147877692, -0.1250369935783831, 0.17692337611818643, -0.12798217728330416, -0.5082824442237116, -0.6577423533211235, -0.5082689896863171, -0.12502734084296815, -0.6837727406761231, -0.11962082826897673, -0.6577544710746848, -0.36558090828216466, -0.7398407750951622, -0.5029200290976654, -0.5028664332017206, -0.09799820703114376, -0.6703519191891821, -0.5029683413958262, -0.6577633532177528, -0.6577500883631121, 0.034228646900603274, -0.657736747408876, -0.5031364441562037, -0.6597610412917574, -0.3577463947170478, -0.5032384985425795, -0.6577634459410923, -0.6579877133492231, -0.5029790624735975, -0.6578060842595743, 0.09695059101921422, -0.11138293380778078, -0.3490508059714521, -0.0974463064424239, -0.2808393443218296, -0.6578989011947097, -0.3113543187205608, 5.6562590143016385, -0.625674290835139, -0.7969058973755277, -0.09786035374532334, -0.09790985036641643, 0.033600013605108676, -0.22905219364638005, 0.0969479904419465, -0.2746443613685292, 0.5185083917361523, -0.0354500440102341, -0.6577762829706706, -0.27449940242443266, -0.6577572956418891]}, {"customdata": [["SHA-256"], ["Ethash"], ["Scrypt"], ["CryptoNight-V7"], ["Ethash"], ["Equihash"], ["Multiple"], ["Scrypt"], ["X11"], ["Scrypt"], ["Multiple"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Quark"], ["Groestl"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["X11"], ["Multiple"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["NeoScrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["HybridScryptHash256"], ["Scrypt"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["SHA-256"], ["SHA-256"], ["SHA-256"], ["SHA-256"], ["SHA-256"], ["X11"], ["Scrypt"], ["Lyra2REv2"], ["Scrypt"], ["SHA-256"], ["CryptoNight"], ["CryptoNight"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Stanford Folding"], ["Multiple"], ["QuBit"], ["Scrypt"], ["Scrypt"], ["M7 POW"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["X11"], ["Lyra2RE"], ["SHA-256"], ["X11"], ["Scrypt"], ["Scrypt"], ["Ethash"], ["Blake2b"], ["X11"], ["SHA-256"], ["Scrypt"], ["1GB AES Pattern Search"], ["Scrypt"], ["SHA-256"], ["X11"], ["Dagger"], ["Scrypt"], ["X11GOST"], ["Scrypt"], ["ScryptOG"], ["X11"], ["Scrypt"], ["X11"], ["Equihash"], ["CryptoNight"], ["SHA-256"], ["Multiple"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["Lyra2Z"], ["Ethash"], ["Equihash"], ["Scrypt"], ["X11"], ["X11"], ["CryptoNight"], ["Scrypt"], ["CryptoNight"], ["Lyra2RE"], ["X11"], ["CryptoNight-V7"], ["Scrypt"], ["X11"], ["Equihash"], ["Scrypt"], ["Lyra2RE"], ["Dagger-Hashimoto"], ["Scrypt"], ["NIST5"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["CryptoNight-V7"], ["Argon2d"], ["Blake2b"], ["Cloverhash"], ["CryptoNight"], ["X11"], ["Scrypt"], ["Scrypt"], ["X11"], ["X11"], ["CryptoNight"], ["Time Travel"], ["Scrypt"], ["Keccak"], ["X11"], ["SHA-256"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["Scrypt"], ["CryptoNight"], ["Equihash"], ["X11"], ["NeoScrypt"], ["Equihash"], ["Dagger"], ["Scrypt"], ["X11"], ["NeoScrypt"], ["Ethash"], ["NeoScrypt"], ["Multiple"], ["CryptoNight"], ["CryptoNight"], ["Ethash"], ["Scrypt"], ["X11"], ["CryptoNight-V7"], ["Scrypt"], ["BLAKE256"], ["X11"], ["NeoScrypt"], ["NeoScrypt"], ["NeoScrypt"], ["Scrypt"], ["X11"], ["SHA-256"], ["C11"], ["Ethash"], ["Scrypt"], ["CryptoNight"], ["SkunkHash"], ["CryptoNight"], ["Scrypt"], ["Dagger"], ["Lyra2REv2"], ["Scrypt"], ["Scrypt"], ["X11"], ["Ethash"], ["CryptoNight"], ["Scrypt"], ["SHA-256"], ["IMesh"], ["Quark"], ["Equihash"], ["Lyra2Z"], ["NeoScrypt"], ["X11"], ["CryptoNight"], ["NIST5"], ["Lyra2RE"], ["Tribus"], ["Lyra2Z"], ["CryptoNight"], ["CryptoNight Heavy"], ["CryptoNight"], ["Jump Consistent Hash"], ["CryptoNight"], ["SHA-512"], ["X16R"], ["HMQ1725"], ["X11"], ["Scrypt"], ["Scrypt"], ["CryptoNight-V7"], ["Cryptonight-GPU"], ["XEVAN"], ["CryptoNight Heavy"], ["CryptoNight"], ["X11"], ["SHA-256"], ["X11"], ["X16R"], ["Equihash"], ["HMQ1725"], ["Lyra2Z"], ["SHA-256"], ["PHI1612"], ["CryptoNight"], ["Blake"], ["Blake"], ["Blake"], ["Blake"], ["Equihash"], ["Exosis"], ["Scrypt"], ["Equihash"], ["Equihash"], ["Lyra2REv2"], ["QuBit"], ["SHA-256"], ["X13"], ["SHA-256"], ["Ethash"], ["Scrypt"], ["NeoScrypt"], ["Blake"], ["Scrypt"], ["SHA-256"], ["Scrypt"], ["Groestl"], ["Scrypt"], ["Scrypt"], ["Multiple"], ["Equihash+Scrypt"], ["Lyra2Z"], ["Ethash"], ["Equihash"], ["CryptoNight"], ["Equihash"]], "hovertemplate": "<b>%{hovertext}</b><br><br>Class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>", "hovertext": ["Bitcoin", "Ethereum", "Litecoin", "Monero", "Ethereum Classic", "ZCash", "DigiByte", "ProsperCoin", "Spreadcoin", "Argentum", "MyriadCoin", "MoonCoin", "ZetaCoin", "SexCoin", "Quatloo", "QuarkCoin", "Riecoin", "Digitalcoin ", "Catcoin", "CannaCoin", "CryptCoin", "Verge", "DevCoin", "EarthCoin", "E-Gulden", "Einsteinium", "Emerald", "Franko", "FeatherCoin", "GrandCoin", "GlobalCoin", "GoldCoin", "Infinite Coin", "IXcoin", "KrugerCoin", "LuckyCoin", "Litebar ", "MegaCoin", "MediterraneanCoin", "MinCoin", "NobleCoin", "Namecoin", "NyanCoin", "RonPaulCoin", "StableCoin", "SmartCoin", "SysCoin", "TigerCoin", "TerraCoin", "UnbreakableCoin", "Unobtanium", "UroCoin", "ViaCoin", "Vertcoin", "WorldCoin", "JouleCoin", "ByteCoin", "DigitalNote ", "MonaCoin", "Gulden", "PesetaCoin", "Wild Beast Coin", "Flo", "ArtByte", "Folding Coin", "Unitus", "CypherPunkCoin", "OmniCron", "GreenCoin", "Cryptonite", "MasterCoin", "SoonCoin", "1Credit", "MarsCoin ", "Crypto", "Anarchists Prime", "BowsCoin", "Song Coin", "BitZeny", "Expanse", "Siacoin", "MindCoin", "I0coin", "Revolution VR", "HOdlcoin", "Gamecredits", "CarpeDiemCoin", "Adzcoin", "SoilCoin", "YoCoin", "SibCoin", "Francs", "Aiden", "BolivarCoin", "Omni", "PizzaCoin", "Komodo", "Karbo", "ZayedCoin", "Circuits of Value", "DopeCoin", "DollarCoin", "Shilling", "ZCoin", "Elementrem", "ZClassic", "KiloCoin", "ArtexCoin", "Kurrent", "Cannabis Industry Coin", "OsmiumCoin", "Bikercoins", "HexxCoin", "PacCoin", "Citadel", "BeaverCoin", "VaultCoin", "Zero", "Canada eCoin", "Zoin", "DubaiCoin", "EB3coin", "Coinonat", "BenjiRolls", "ILCoin", "EquiTrader", "Quantum Resistant Ledger", "Dynamic", "Nano", "ChanCoin", "Dinastycoin", "DigitalPrice", "Unify", "SocialCoin", "ArcticCoin", "DAS", "LeviarCoin", "Bitcore", "gCn Coin", "SmartCash", "Onix", "Bitcoin Cash", "Sojourn Coin", "NewYorkCoin", "FrazCoin", "Kronecoin", "AdCoin", "Linx", "Sumokoin", "BitcoinZ", "Elements", "VIVO Coin", "Bitcoin Gold", "Pirl", "eBoost", "Pura", "Innova", "Ellaism", "GoByte", "SHIELD", "UltraNote", "BitCoal", "DaxxCoin", "BoxyCoin", "AC3", "Lethean", "PopularCoin", "Photon", "Sucre", "SparksPay", "GoaCoin", "GunCoin", "IrishCoin", "Pioneer Coin", "UnitedBitcoin", "Interzone", "1717 Masonic Commemorative Token", "Crypto Wisdom Coin", "TurtleCoin", "MUNcoin", "Niobio Cash", "ShareChain", "Travelflex", "KREDS", "BitFlip", "LottoCoin", "Crypto Improvement Fund", "Callisto Network", "BitTube", "Poseidon", "Manna", "Aidos Kuneen", "Cosmo", "Bitrolium", "Alpenschillling", "Rapture", "FuturoCoin", "Monero Classic", "Jumpcoin", "Infinex", "KEYCO", "GINcoin", "PlatinCoin", "Loki", "Newton Coin", "MassGrid", "PluraCoin", "Arionum", "Motion", "PlusOneCoin", "Axe", "HexCoin", "Deimos", "Webchain", "Ryo", "Urals Coin", "Qwertycoin", "Bitcoin Nova", "DACash", "Project Pai", "Azart", "Xchange", "CrypticCoin", "Brazio", "Actinium", "Bitcoin SV", "Argoneum", "FREDEnergy", "BlakeBitcoin", "Universal Molecule", "Lithium", "Electron", "PirateCash", "Exosis", "Block-Logic", "Beam", "Bithereum", "Scribe Network", "SLICE", "BLAST", "Bitcoin Rhodium", "GlobalToken", "Media Protocol Token", "SolarCoin", "UFO Coin", "BlakeCoin", "Crypto Escudo", "Crown Coin", "SmileyCoin", "Groestlcoin", "Bata", "Pakcoin", "JoinCoin", "Vollar", "TecraCoin", "Reality Clash", "ChainZilla", "Beldex", "Horizen"], "legendgroup": "0", "marker": {"color": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "coloraxis": "coloraxis", "symbol": "diamond"}, "mode": "markers", "name": "0", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [-0.14028129635507744, -0.1344931599319322, -0.16802763126752862, -0.12980729551137088, -0.13287508323523, -0.14375420897848418, 0.18621387218460397, -0.16922051674649144, -0.21518813315100258, -0.1688220334864016, -0.07736937896538418, 2.6571934762368805, -0.13724730195354995, -0.16596667605876034, -0.16861875766400777, -0.21396745171992562, -0.195235461013558, -0.16866820897064583, -0.169202035243164, -0.16929192627217834, -0.21528118837711366, 0.2110821357836844, 0.2529354771258124, 0.08956489337230461, -0.1690277356424271, -0.16445520403167233, -0.16896476242006053, -0.1693512814962281, -0.17434257419561994, -0.14773519857582942, -0.16809715805626552, -0.1683863602679255, 1.65087365750023, -0.14024103743295363, -0.16556662090095903, -0.16905526526971842, -0.16942454016164665, -0.1686536448450762, -0.1469068967125468, -0.16930103127007154, -0.028909482229689094, -0.14032191940816877, -0.16270654548958743, -0.1692798979682355, -0.16729998238485144, -0.16874580756099228, -0.12695544041768395, -0.13976377449617117, -0.14006295267338503, -0.14004622868375266, -0.1406597183383537, -0.21546193512915512, -0.16898448993571594, -0.1669439227145783, -0.16597199790310446, -0.13983385249395905, 4.005232136121385, 0.46547010151314455, -0.16780653006260648, -0.15179803223854496, -0.16647296914093831, -0.16942687651101848, -0.16633215021563585, -0.15199737674665084, -0.11330939869763113, -0.11275633975000979, -0.1732450462457139, -0.16929462383411367, -0.037141948250896166, -0.11889785160545309, -0.1694360905973466, -0.1403508678252951, 0.5077689011514044, -0.2148241653731773, -0.1508931180498332, -0.14008063401228235, -0.21512075493683896, -0.16748634886108646, -0.16664559677787544, -0.13560606309792622, 0.40959799736258784, -0.2151575110599773, -0.1402422070944233, -0.1652291222463856, -0.13663581822816664, -0.1679410977229913, 0.2953083616538038, -0.21428466007776892, -0.1534427730807549, -0.16820119255036214, -0.13012107620043648, -0.16921766086963733, -0.13908240722967424, -0.21511944403805833, -0.16943615147747998, -0.21527573688942045, -0.23859528024701035, 0.30410226012837915, -0.14051295647283274, -0.09202066101910124, -0.1664887239710619, -0.14046986620271246, -0.16908614296940822, -0.0773187216806739, -0.1353375979608369, -0.14377673728158824, -0.093338919119255, -0.20130610359865658, -0.21301772681862363, 0.3040960942361808, -0.16941717523524596, 0.3042322141519122, -0.15145486263830304, 0.5274639218697524, -0.12852484172500328, -0.16938414269701751, -0.2077294083188324, -0.1437895274669249, -0.1674412857368858, -0.15116166352293323, -0.132552162082915, -0.13918831691608768, -0.17311610315256185, -0.1689289268421358, -0.10548472033382096, -0.16874446035929644, -0.12837445474612474, -0.15463091847067634, -0.0075885925121333955, -0.13844885022744927, 0.3416882143761288, -0.2142938326232041, -0.1690757789271398, -0.16883066586858833, -0.21470408714578706, -0.21530478906932743, 0.3045068552912868, -0.13073285398153206, 3.378681547511223, -0.12061578151987655, -0.20582086716482592, -0.14028042534040838, -0.09215136744293308, 1.6512274855442897, -0.1691777703900298, -0.1686079987138958, -0.16829679394023553, -0.1682831739461839, 0.3046972778385949, 0.07692959964110518, -0.16781626762902535, -0.179220750079438, -0.14362919541852262, -0.15215291633370046, -0.1674394193995737, -0.21067019516610175, -0.1790601238477174, -0.133640631902286, -0.17916420221153903, -0.10259392510410648, 1.1751027497891702, 0.30407836350456974, -0.05562197753692131, -0.16863908057856872, -0.21040619181647524, -0.11705132390944181, -0.08217344595394968, 1.154255747486702, -0.21527771552146294, -0.17921804632805746, -0.17918628523806027, -0.1719751639561579, -0.1683983672449162, -0.21520006361306623, -0.140258932077382, -0.1728287531546173, -0.2488636984451618, -0.16904741060541895, 8.341520802142368, -0.15419941070502574, 0.3081075195210757, 0.03147599512794071, -0.1516336933141013, -0.15192018517355624, -0.16899563919721275, 0.15053709759111347, -0.2091066663333663, -0.08486487187235935, 0.3123092433995986, -0.1692421359202476, -0.03667273052836301, -0.1479556159249865, -0.21888894331607475, -0.14257905879636612, -0.07501620583985442, -0.17926093227456852, -0.21433632386775264, 0.3042684195567652, -0.1731773465670062, -0.15129401001088255, -0.18472337087027518, -0.07740636526236669, 0.30834676430483865, 0.613337058804162, 2.184099004389037, -0.15833054987508244, 0.31842078784727407, -0.18297130849951607, -0.1872955650167466, -0.14968164102689538, -0.2152591555955753, -0.1692677834391034, -0.15866090952072384, -0.1169461003175875, -0.13142292331677677, -0.1596493970527051, 3.237332475517091, 0.31745788927458946, -0.21476920739152014, -0.09846996465254902, -0.21522942048670582, -0.18670123527801707, -0.03413990566736803, -0.1474603174411282, -0.07678068419235284, -0.14028046448948273, -0.19494299379181515, 0.3871765523696956, -0.13040137942912433, -0.13002428388572407, -0.13043070946026966, -0.07898329094434275, -0.2560903874833574, -0.1339279120157017, -0.16811598772896336, -0.1416139522067414, -0.1434552294623266, -0.16789615276715236, -0.17242541709456105, -0.1395272354040362, -0.09459746028807095, -0.1383502168507744, 0.0650604010298079, 0.5533775857789044, -0.10133473029990309, -0.07899347774789656, -0.15215431106357138, -0.14006229658473401, 0.5752495974224516, -0.19477851298162732, -0.1693474060530544, -0.1672216457745697, -0.11352227993442457, -0.11707419066290027, -0.07550639383410652, -0.1353721086568668, -0.24093268805266335, 0.3267157755716168, -0.14375531048384688], "y": [-1.3484900371649857, -2.0551071940608936, -0.9991935644429482, -2.2039761412789316, -2.0551875020387382, -2.1205037718660207, -1.853217538412918, -0.9991959248928507, -0.9899396142822386, -0.9992098576125179, -1.8503332929861467, -1.1509796182299081, -1.3484763633946752, -0.9992278096848917, -0.999226395706806, -0.587568041291091, -1.2589501291641196, -0.9991934888805256, -0.9991952338479805, -0.9991933246703592, -0.9899417526800126, -1.8493072122367586, -1.347854098873497, -0.9985576119339988, -0.9991887165835146, -0.9992046809587202, -0.9991937376901237, -0.999194267230705, -1.6882153069973993, -0.9993310337647863, -0.9991867768140239, -0.9991990962131218, -0.9918757561503369, -1.348488531837125, -0.9992231911527963, -0.9991890754256535, -0.9991903777963106, -0.9991888098686538, -2.1942918629569874, -0.9991915596758627, -1.0039932902973616, -1.3484915561080975, -0.9991642450828327, -0.9991981452256842, -0.9992776635566423, -0.9991984247474704, -1.348573111025323, -1.3484881280339456, -1.3484959540156167, -1.3485208085447382, -1.348490273458551, -0.989936441604288, -0.9991884406215091, -1.5243367689099352, -0.999238348669625, -1.3484893992616576, -2.280488330871967, -2.295859783010462, -0.9991994586493145, -0.9996568749472252, -0.9991906754950896, -0.9991913220829999, -0.9991811281546797, -0.9992083733069869, -2.2759363206808305, -1.850315407355846, -1.622124071967321, -0.9991868754260923, -1.0009484987990287, -2.3374348550315402, -0.9991903198490084, -1.3484926385230283, -1.0355565019781412, -0.9899347218980932, -2.1834454728359414, -1.3485045004870966, -0.989937765455165, -0.9992579720373191, -0.9992531953189334, -2.0551601429237083, -2.2068653434942918, -0.9899357871963275, -1.3484885755721256, -0.9991734107877108, -2.1583854102678997, -0.999190328853242, -1.3466896624073548, -0.9899487458208468, -2.266628430934217, -0.9992566135532095, -2.3270488702735728, -0.9991951475857063, -2.2445061918794895, -0.9899403985295071, -0.999190320093704, -0.9899462425007048, -1.6755757082323561, -2.295194993279417, -1.3484911464392935, -1.850344703361047, -0.9992138035329049, -1.3484901402552523, -0.9991969352046245, -1.9383632838105123, -2.055156339985296, -2.120504614225432, -1.0030497687949935, -0.9897423999220235, -0.9899979289684551, -2.295202599577804, -0.9991910171977462, -2.295200191986463, -2.1834290692657614, -1.0292098403811714, -2.204052235522293, -0.9991902150386097, -0.9903178360403995, -2.120502410374943, -0.9991823686641855, -2.183425481959566, -2.3275773337211048, -1.0007409954635866, -0.7305224672603212, -0.9991947582639876, -1.3488511782889099, -0.9992123212439581, -2.2039929705711283, -2.1956842511104684, -2.222692601536277, -2.348283239218646, -2.2951239507647645, -0.9899598171544507, -0.9991893575331857, -0.9992175561358287, -0.989948336143734, -0.9899432386076324, -2.295209367990535, -2.429683937576579, -1.000626184449725, -1.2712535638037592, -0.990313525729724, -1.3484900045967219, -1.003340626155076, -0.9311131792320502, -0.9991936560343535, -0.9992152650556887, -0.9992143571093142, -0.9992138478419212, -2.2952256416580035, -2.126319041473758, -0.9893618554373517, -1.6881905161770998, -2.1204990974638473, -2.266664893210542, -0.9991822988796661, -0.9899919557903237, -1.6881965795824438, -2.055263063373814, -1.6881916202951006, -1.8503779677827945, -2.3196149887096422, -2.2951975631084123, -2.058863334891548, -0.99922715560418, -0.9901161889170159, -2.2041693556258872, -0.9992796658251698, -1.2881120773980241, -0.9899428297665738, -1.6881863919451416, -1.6881925801096238, -1.688236752484703, -0.999194016239793, -0.9899420719438462, -1.348488641722817, -1.2493667013587522, -0.20342270986373592, -0.9991914638210045, -2.6651824682808063, -2.278541085545477, -2.2952638223450106, -0.9983827942178246, -2.2666077239225344, -1.5244562648718456, -0.9992002563894149, -0.9995680312615863, -0.9900340720205875, -2.0576099315936127, -2.295551942011599, -0.9991967332588607, -1.3513370797669713, -2.2090575156399357, -0.5875944163509754, -2.1204926872431704, -1.9384639976638316, -1.6881879955004808, -0.9899614059514247, -2.295194412775224, -0.7305064841410124, -2.1834339709077177, -1.2001827404352556, -1.9383592522095445, -2.295431895016479, -2.2789154651832932, -2.348269183770255, -2.3993269682340297, -2.2953234242680534, -0.9196033116354797, -1.2457717561735395, -2.300269229653929, -0.9899429404165039, -0.9991984333531848, -0.9994575305350177, -2.204668660793184, -2.0761588631186303, -0.9417260679559488, -2.3043919538450033, -2.2953594281637923, -0.9899440658394142, -1.3483206793915596, -0.9899445106766251, -1.2458017834981319, -2.1214870565107553, -2.300310888987882, -1.9383851406783497, -1.348490006060551, -0.822481915838802, -2.29749986634887, -2.192884314426852, -2.192926618748049, -2.1928882466167128, -2.1956413095651635, -0.2687984595559563, -2.0671673947669147, -0.9992210070069737, -2.1205858775429416, -2.1204992214610874, -1.5243379760883256, -1.6221604773408618, -1.348490674421882, -0.341825966084879, -1.3485163986810522, -2.0543508806221196, -1.0379413281137086, -1.6879422537171458, -2.1956416904615765, -0.9992142412633267, -1.3484959294837005, -1.0048713715035418, -1.2589471243210455, -0.9991899410698853, -0.9992291389235329, -1.8503459242119997, -2.2076654815098213, -1.9384219793288153, -2.0551564786940424, -1.803221633510487, -2.295281623347026, -2.1205038130525833], "z": [0.23190176398938814, 0.36716010018918666, -0.07686255435946705, 0.41075678992167797, 0.3671262956013683, 0.5642834699544013, 0.32993583514256236, -0.07682540490136, 0.07805375158316034, -0.07683489079944099, 0.33746803102403694, -0.13358952834637663, 0.23180575433996975, -0.07691912604632986, -0.07683785669641714, 0.07249260184448378, 0.10088025427273678, -0.07684287192196407, -0.07682611045108732, -0.07682372554029646, 0.07805703770545784, 0.32839449431531076, 0.219684236348725, -0.08490970670095423, -0.07683276451128997, -0.07697019679815582, -0.07683370363343651, -0.0768217131889618, 0.2694165965567958, -0.07745923292735242, -0.07686176457835335, -0.07685042492217631, -0.1342462985711342, 0.23190022706507663, -0.0769323448585085, -0.0768318467170983, -0.07682023302613387, -0.07684424908910506, 0.38132695533899047, -0.07682379612263464, -0.08018706489071373, 0.23190331481473364, -0.07703199839743156, -0.07682313796430033, -0.07686822575439047, -0.07683950546083979, 0.23150850036839038, 0.23188547126555475, 0.2318962252122362, 0.23190064746986844, 0.23191344720867135, 0.07806154071243881, -0.07683414910576206, 0.22564375400915765, -0.0769168691892261, 0.23188787861550417, 0.28603370691815383, 0.39793264631926134, -0.07686818242658867, -0.07726958547953745, -0.07691093323760101, -0.07681997363392554, -0.07691715960585696, -0.077352535424113, 0.5181235160691788, 0.33855261007276294, 0.23706651597631215, -0.0768249235159629, -0.08053866555958972, 0.593540651281356, -0.07681988936525992, 0.23190441994928082, -0.09042069297580627, 0.07804158804849579, 0.346047532181473, 0.23189846637224404, 0.07805131252962577, -0.07686640613266929, -0.0768932075742351, 0.36720483790904895, 0.3565649994463971, 0.0780520498478694, 0.2319002717180638, -0.07695260996075876, 0.38288008136408896, -0.0768658578633855, 0.21815001530797115, 0.07802778390745502, 0.4338810262365005, -0.0768446948620969, 0.3861270342875648, -0.07682564710393815, 0.3387935951516632, 0.07805179519134632, -0.07681988744462558, 0.0780577618260247, 1.0593823739812176, 0.40276258908850454, 0.2319091077424937, 0.33792081693738446, -0.07690585518156538, 0.2319075828936975, -0.07682933616381936, 0.3720259711150058, 0.36719582741460083, 0.5642843299947234, -0.07839329043660946, 0.07758771612075169, 0.07799859491016112, 0.40276428942065806, -0.07682033249847388, 0.4027596256143951, 0.3460615475331862, 0.06301727352535676, 0.41073246865532115, -0.07682150755437218, 0.07789952047458275, 0.564284285566157, -0.07688280785555272, 0.34605181930927165, 0.39690881639334497, -0.07744200405600346, -0.014991767792955654, -0.07683460285711793, 0.23090351507799886, -0.07683678681737408, 0.4107160733515412, 0.5344975948921804, 0.3725368291223083, 0.42387596594502147, 0.4015927295159703, 0.07803026490480673, -0.0768311599012219, -0.07683309630658476, 0.07804059971244996, 0.07805805854428127, 0.40275300303911465, 0.32260789710802357, -0.18563755412567384, 0.11298298140661762, 0.07783997768687895, 0.23190173073752565, -0.07837203807740249, -0.14632561039372155, -0.07682716996257745, -0.07684039825634134, -0.07685014798103249, -0.07685066793781957, 0.40275037985449924, 0.5586525594389765, 0.07648233804233597, 0.2695616741779402, 0.5642786974377056, 0.4338486057823238, -0.07688287910483149, 0.07792522311080062, 0.2695579392917867, 0.3671648435465148, 0.26956015465487343, 0.3382525463963027, 0.3808299642710524, 0.40276383430744317, 0.36548088099864967, -0.07683708084898874, 0.07794177985008578, 0.41040292564454706, -0.07948542668119846, 0.08353791366608268, 0.07805714484370961, 0.2695607718996753, 0.2695610243316512, 0.2693480592917674, -0.07685106467858227, 0.07805460657032404, 0.23190079914156086, 0.018807028651891138, 0.16953171724381638, -0.07683161387011911, 0.2291014750872001, 0.4976300938628825, 0.4026530999235126, -0.08315822363256609, 0.43382128518658053, 0.22520551539345882, -0.0768314594686675, -0.08658388598037353, 0.0778855103430685, 0.3661311392460673, 0.40258112423803016, -0.07682457956795069, 0.22928132094486853, 0.3458517444403915, 0.07264917377461777, 0.5642451330858137, 0.3719751733016145, 0.26956240911323603, 0.07803188705022822, 0.40275736447128063, -0.01499305909478626, 0.3460575749404554, 0.07377599869538264, 0.372027865370263, 0.40267912521991595, 0.35067825639947575, 0.3554950190081179, 0.6545330526304408, 0.40234780996470865, 0.15410016318858752, 0.20047604616708659, 0.37794801946380774, 0.07805659611209785, -0.07682345325325132, -0.0770981484487341, 0.4104988601636375, 0.2957531176338083, 0.21546405812439942, 0.2750517470766231, 0.40238456957313384, 0.07804175397598928, 0.2305824491614291, 0.07805599365139472, 0.20046373473648124, 0.5611081809599124, 0.3778879890879367, 0.3720137678580454, 0.23190173223208013, 0.1022452940394777, 0.40066588108324996, 0.40502287755692673, 0.40501968438000796, 0.4050245604373661, 0.40398938081989366, 0.3665883946511746, 0.26089000670602963, -0.07685438690463772, 0.5642339656414114, 0.5642733726990323, 0.22567327437432555, 0.23704854348476354, 0.23187870356012827, -0.04476251431238165, 0.23184762008012544, 0.36087371145270325, -0.09134947222777741, 0.26711740890507385, 0.40398976971122674, -0.07734654430085026, 0.23189620016540094, -0.09859025462471871, 0.10086560653777876, -0.07682269160250799, -0.07688027234411252, 0.33858222349389616, 0.3869666285658543, 0.37198190081657384, 0.3671969161511235, 1.026003932462419, 0.40208444088250067, 0.5642835120054592]}, {"customdata": [["VeChainThor Authority"], ["SHA-256 + Hive"], ["Proof-of-Authority"], ["ECC 256K1"], ["Leased POS"]], "hovertemplate": "<b>%{hovertext}</b><br><br>Class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>", "hovertext": ["Vechain", "LitecoinCash", "Poa Network", "Acute Angle Cloud", "Waves"], "legendgroup": "1", "marker": {"color": [1, 1, 1, 1, 1], "coloraxis": "coloraxis", "symbol": "square"}, "mode": "markers", "name": "1", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [4.6613708088394565, -0.3603968950875864, -0.4055930330411888, -0.3547714072392418, -0.44889404903199204], "y": [3.996766128137127, 3.934845818132301, 1.030032717176579, 3.1644948403985915, 3.1646892093773604], "z": [10.12698509085421, 16.862323150616362, 14.198472351064767, 18.344108291306497, 9.68983401279352]}, {"customdata": [["TRC10"]], "hovertemplate": "<b>%{hovertext}</b><br><br>Class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>", "hovertext": ["BitTorrent"], "legendgroup": "2", "marker": {"color": [2], "coloraxis": "coloraxis", "symbol": "x"}, "mode": "markers", "name": "2", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [35.4886124109449], "y": [1.6943937575355463], "z": [-1.3688182507984774]}],
                        {"coloraxis": {"colorbar": {"title": {"text": "Class"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "legend": {"title": {"text": "Class"}, "tracegroupgap": 0, "x": 0, "y": 1}, "margin": {"t": 60}, "scene": {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "xaxis": {"title": {"text": "PC 1"}}, "yaxis": {"title": {"text": "PC 2"}}, "zaxis": {"title": {"text": "PC 3"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 800},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('01225515-950e-4ee9-a340-ac7008e2b7a1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


#### Table of Tradable Cryptocurrencies


```python
# Table with tradable cryptos
clustered_df[["CoinName", "Algorithm", "ProofType", "TotalCoinSupply", "TotalCoinsMined", "Class"]].hvplot.table()
```




<div id='1203'>





  <div class="bk-root" id="3599fbd4-cfa4-474a-bee3-90708add65bf" data-root-id="1203"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"fa1300db-27a8-4419-b6e1-bcb4bf921d60":{"roots":{"references":[{"attributes":{"editor":{"id":"1218"},"field":"ProofType","formatter":{"id":"1217"},"title":"ProofType"},"id":"1219","type":"TableColumn"},{"attributes":{},"id":"1223","type":"StringEditor"},{"attributes":{},"id":"1217","type":"StringFormatter"},{"attributes":{"source":{"id":"1205"}},"id":"1238","type":"CDSView"},{"attributes":{"children":[{"id":"1204"},{"id":"1237"},{"id":"1243"}],"margin":[0,0,0,0],"name":"Row01864","tags":["embedded"]},"id":"1203","type":"Row"},{"attributes":{},"id":"1233","type":"IntEditor"},{"attributes":{},"id":"1212","type":"StringFormatter"},{"attributes":{"format":"0,0.0[00000]"},"id":"1227","type":"NumberFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01869","sizing_mode":"stretch_width"},"id":"1243","type":"Spacer"},{"attributes":{},"id":"1208","type":"StringEditor"},{"attributes":{},"id":"1232","type":"NumberFormatter"},{"attributes":{},"id":"1207","type":"StringFormatter"},{"attributes":{"editor":{"id":"1213"},"field":"Algorithm","formatter":{"id":"1212"},"title":"Algorithm"},"id":"1214","type":"TableColumn"},{"attributes":{},"id":"1213","type":"StringEditor"},{"attributes":{"columns":[{"id":"1209"},{"id":"1214"},{"id":"1219"},{"id":"1224"},{"id":"1229"},{"id":"1234"}],"height":300,"reorderable":false,"source":{"id":"1205"},"view":{"id":"1238"},"width":700},"id":"1237","type":"DataTable"},{"attributes":{"editor":{"id":"1228"},"field":"TotalCoinsMined","formatter":{"id":"1227"},"title":"TotalCoinsMined"},"id":"1229","type":"TableColumn"},{"attributes":{},"id":"1222","type":"StringFormatter"},{"attributes":{},"id":"1228","type":"NumberEditor"},{"attributes":{},"id":"1206","type":"Selection"},{"attributes":{},"id":"1240","type":"UnionRenderers"},{"attributes":{"editor":{"id":"1223"},"field":"TotalCoinSupply","formatter":{"id":"1222"},"title":"TotalCoinSupply"},"id":"1224","type":"TableColumn"},{"attributes":{},"id":"1218","type":"StringEditor"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01868","sizing_mode":"stretch_width"},"id":"1204","type":"Spacer"},{"attributes":{"editor":{"id":"1233"},"field":"Class","formatter":{"id":"1232"},"title":"Class"},"id":"1234","type":"TableColumn"},{"attributes":{"data":{"Algorithm":["Scrypt","Scrypt","X13","SHA-256","Ethash","Scrypt","X11","CryptoNight-V7","Ethash","Equihash","SHA-512","Multiple","SHA-256","SHA-256","Scrypt","X15","X11","Scrypt","Scrypt","Scrypt","Multiple","Scrypt","SHA-256","Scrypt","Scrypt","Scrypt","Quark","Groestl","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","Groestl","Multiple","SHA-256","Scrypt","Scrypt","Scrypt","Scrypt","PoS","Scrypt","Scrypt","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","X11","SHA-256","Scrypt","Scrypt","Scrypt","SHA3","Scrypt","HybridScryptHash256","Scrypt","Scrypt","SHA-256","Scrypt","X13","Scrypt","SHA-256","Scrypt","X13","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","X11","SHA-256","Multiple","SHA-256","PHI1612","X11","SHA-256","SHA-256","SHA-256","X11","Scrypt","Scrypt","Scrypt","Scrypt","Lyra2REv2","Scrypt","X11","Multiple","SHA-256","X13","Scrypt","CryptoNight","CryptoNight","Shabal256","Counterparty","Scrypt","SHA-256","Groestl","Scrypt","Scrypt","Scrypt","X13","Scrypt","Scrypt","Scrypt","Scrypt","X13","Scrypt","Stanford Folding","X11","Multiple","QuBit","Scrypt","Scrypt","Scrypt","M7 POW","Scrypt","SHA-256","Scrypt","X11","SHA3","X11","Lyra2RE","SHA-256","QUAIT","X11","X11","Scrypt","Scrypt","Scrypt","Ethash","X13","Blake2b","SHA-256","X15","X11","SHA-256","BLAKE256","Scrypt","1GB AES Pattern Search","SHA-256","X11","Scrypt","SHA-256","SHA-256","NIST5","Scrypt","Scrypt","X11","Dagger","Scrypt","X11GOST","X11","Scrypt","SHA-256","Scrypt","PoS","Scrypt","ScryptOG","X11","X11","SHA-256","SHA-256","NIST5","X11","Scrypt","POS 3.0","Scrypt","Scrypt","Scrypt","X13","X11","X11","Equihash","X11","Scrypt","CryptoNight","SHA-256","SHA-256","X11","Scrypt","Multiple","Scrypt","Scrypt","Scrypt","SHA-256","Scrypt","Scrypt","SHA-256D","PoS","Scrypt","X11","Lyra2Z","PoS","X13","X14","PoS","SHA-256D","Ethash","Equihash","DPoS","X11","Scrypt","X11","X13","X11","PoS","Scrypt","Scrypt","X11","PoS","X11","SHA-256","Scrypt","X11","Scrypt","Scrypt","X11","CryptoNight","Scrypt","Scrypt","Scrypt","Scrypt","Quark","QuBit","Scrypt","CryptoNight","Lyra2RE","Scrypt","SHA-256","X11","Scrypt","X11","Scrypt","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X13","X11","Equihash","Scrypt","Scrypt","Lyra2RE","Scrypt","Dagger-Hashimoto","X11","Blake2S","X11","Scrypt","PoS","X11","NIST5","PoS","X11","Scrypt","Scrypt","Scrypt","SHA-256","X11","Scrypt","Scrypt","SHA-256","PoS","Scrypt","X15","SHA-256","Scrypt","POS 3.0","CryptoNight-V7","536","Argon2d","Blake2b","Cloverhash","CryptoNight","NIST5","X11","NIST5","Skein","Scrypt","X13","Scrypt","X11","X11","Scrypt","CryptoNight","X13","Time Travel","Scrypt","Keccak","SkunkHash v2 Raptor","X11","Skein","SHA-256","X11","Scrypt","VeChainThor Authority","Scrypt","PoS","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","CryptoNight","SHA-512","Ouroboros","X11","Equihash","NeoScrypt","X11","Scrypt","NeoScrypt","Lyra2REv2","Equihash","Scrypt","SHA-256","NIST5","PHI1612","Dagger","Scrypt","Quark","Scrypt","POS 2.0","Scrypt","SHA-256","X11","NeoScrypt","Ethash","NeoScrypt","Quark","X11","DPoS","NIST5","X13","Multiple","Scrypt","CryptoNight","CryptoNight","Ethash","NIST5","Scrypt","Quark","X11","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X11","BLAKE256","X11","Scrypt","Scrypt","NeoScrypt","NeoScrypt","Quark","NeoScrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","X11","SHA-256","C11","POS 3.0","Ethash","Scrypt","Scrypt","CryptoNight","SkunkHash","Scrypt","CryptoNight","X11","Scrypt","SHA-256","Dagger","Lyra2REv2","X13","Proof-of-BibleHash","SHA-256 + Hive","Scrypt","Scrypt","Skein","X11","Skein","X11","C11","Proof-of-Authority","X11","XEVAN","Scrypt","VBFT","Ethash","CryptoNight","Scrypt","SHA-256","IMesh","PHI1612","Quark","NIST5","Scrypt","Scrypt","Equihash","Scrypt","Lyra2Z","Green Protocol","PoS","Scrypt","NeoScrypt","Semux BFT consensus","X11","Quark","PoS","CryptoNight","X16R","Scrypt","NIST5","Lyra2RE","XEVAN","Tribus","Scrypt","Lyra2Z","Scrypt","Scrypt","CryptoNight","X11","SHA-256D","CryptoNight Heavy","CryptoNight","Scrypt","Scrypt","X11","Scrypt","Jump Consistent Hash","SHA-256D","CryptoNight","Scrypt","X15","Scrypt","Quark","Scrypt","SHA-512","SHA-256","DPoS","X16R","HMQ1725","X11","Scrypt","X16R","Quark","Quark","Scrypt","Lyra2REv2","Quark","Scrypt","Scrypt","Scrypt","CryptoNight-V7","Cryptonight-GPU","XEVAN","CryptoNight Heavy","CryptoNight","X11","X11","X11","Scrypt","PoS","SHA-256","Keccak","X11","X11","Tribus","Scrypt","NeoScrypt","SHA-512","X16R","ECC 256K1","Equihash","XEVAN","HMQ1725","Lyra2Z","SHA-512","SHA-256","Quark","PHI1612","XEVAN","X11","CryptoNight","Quark","Blake","Blake","Blake","Blake","Equihash","Exosis","Scrypt","Scrypt","Equihash","Quark","Equihash","Quark","Scrypt","Lyra2REv2","QuBit","X11","Scrypt","XEVAN","SHA-256D","X11","SHA-256","X13","SHA-256","X11","DPoS","Ethash","Scrypt","Scrypt","X11","NeoScrypt","Scrypt","Blake","Scrypt","SHA-256","Scrypt","X11","Scrypt","Scrypt","SHA-256","X11","SHA-256","Scrypt","Scrypt","Scrypt","Groestl","X11","Scrypt","PoS","Scrypt","Scrypt","X11","SHA-256","DPoS","Scrypt","Scrypt","NeoScrypt","SHA3-256","Multiple","X13","Equihash+Scrypt","Lyra2Z","PHI2","DPoS","Ethash","DPoS","SHA-256","Lyra2Z","Leased POS","PoS","Scrypt","TRC10","Equihash","PoS","SHA-256","Scrypt","CryptoNight","Equihash","Scrypt","Scrypt"],"Class":{"__ndarray__":"AwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAQAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAABAAAAAAAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAEAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAQAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAABAAAAAwAAAAMAAAACAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAA=","dtype":"int32","order":"little","shape":[578]},"CoinName":["42 Coin","404Coin","EliteCoin","Bitcoin","Ethereum","Litecoin","Dash","Monero","Ethereum Classic","ZCash","Bitshares","DigiByte","BitcoinDark","PayCoin","ProsperCoin","KoboCoin","Spreadcoin","Argentum","Aurora Coin","BlueCoin","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","EnergyCoin","QuarkCoin","Riecoin","Digitalcoin ","BitBar","Catcoin","CryptoBullion","CannaCoin","CryptCoin","CasinoCoin","Diamond","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Exclusive Coin","FlutterCoin","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","HoboNickels","HyperStake","Infinite Coin","IOCoin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MaxCoin","MegaCoin","MediterraneanCoin","MintCoin","MinCoin","MazaCoin","Nautilus Coin","NavCoin","NobleCoin","Namecoin","NyanCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","RonPaulCoin","StableCoin","SmartCoin","SuperCoin","SyncCoin","SysCoin","TeslaCoin","TigerCoin","TittieCoin","TorCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","UnitaryStatus Dollar","UltraCoin","ViaCoin","VeriCoin","Vertcoin","WorldCoin","X11 Coin","Crypti","JouleCoin","StealthCoin","ZCC Coin","ByteCoin","DigitalNote ","BurstCoin","StorjCoin","MonaCoin","Neutron","FairCoin","Gulden","RubyCoin","PesetaCoin","Kore","Wild Beast Coin","Dnotes","Flo","8BIT Coin","Sativa Coin","ArtByte","Folding Coin","Ucoin","Unitus","CypherPunkCoin","OmniCron","Vtorrent","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","IslaCoin","Nexus","MarsCoin ","Crypto","Anarchists Prime","Droidz","BowsCoin","Squall Coin","Song Coin","BitZeny","Diggits","Expanse","Paycon","Siacoin","Emercoin","EverGreenCoin","MindCoin","I0coin","Decred","Revolution VR","HOdlcoin","EDRCoin","Hitcoin","Gamecredits","DubaiCoin","CarpeDiemCoin","PWR Coin","BillaryCoin","GPU Coin","Adzcoin","SoilCoin","YoCoin","SibCoin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Francs","Aiden","BolivarCoin","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Omni","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","PizzaCoin","Lutetium Coin","Komodo","GoldBlocks","CarterCoin","Karbo","BitTokens","ZayedCoin","MustangCoin","ZoneCoin","Circuits of Value","RootCoin","DopeCoin","BitCurrency","DollarCoin","Swiscoin","Shilling","BuzzCoin","Opair","PesoBit","Halloween Coin","ZCoin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","Elementrem","ZClassic","ARK","InsaneCoin","KiloCoin","ArtexCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Kurrent","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Cannabis Industry Coin","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","OsmiumCoin","Bikercoins","HexxCoin","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","PacCoin","Impeach","Citadel","Zilbercoin","FirstCoin","BeaverCoin","FindCoin","VaultCoin","Zero","OpenChat","Canada eCoin","Zoin","RenosCoin","DubaiCoin","VirtacoinPlus","TajCoin","Impact","EB3coin","Atmos","HappyCoin","Coinonat","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","BenjiRolls","GrowthCoin","ILCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","EquiTrader","QTUM","Quantum Resistant Ledger","Espers","Dynamic","Nano","ChanCoin","Dinastycoin","Denarius","DigitalPrice","Virta Unique Coin","Bitcoin Planet","Unify","BritCoin","SocialCoin","ArcticCoin","DAS","Linda","LeviarCoin","DeepOnion","Bitcore","gCn Coin","SmartCash","Signatum","Onix","Cream","Bitcoin Cash","Monoeci","Draftcoin","Vechain","Sojourn Coin","Stakecoin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","CoinonatX","Ethereum Dark","Sumokoin","Obsidian","Cardano","Regalcoin","BitcoinZ","TrezarCoin","Elements","TerraNovaCoin","VIVO Coin","Rupee","Bitcoin Gold","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Pirl","Xios","Bitcloud 2.0","eBoost","KekCoin","BlackholeCoin","Infinity Economics","Pura","Innova","Ellaism","GoByte","Alqo","Magnet","Lamden Tau","Electra","Bitcoin Diamond","SHIELD","Cash & Back Coin","UltraNote","BitCoal","DaxxCoin","Bulwark","BoxyCoin","Kalkulus","AC3","Lethean","GermanCoin","LiteCoin Ultra","PopularCoin","PhantomX","Photon","Sucre","Accolade","OmiseGO Classic","SparksPay","GoaCoin","Digiwage","GunCoin","IrishCoin","Trollcoin","Litecoin Plus","Monkey Project","ECC","Pioneer Coin","UnitedBitcoin","Interzone","TokenPay","1717 Masonic Commemorative Token","Crypto Wisdom Coin","My Big Coin","TurtleCoin","MUNcoin","Unified Society USDEX","Niobio Cash","BitSoar Coin","ShareChain","Credence Coin","Travelflex","KREDS","Tokyo Coin","BiblePay","LitecoinCash","BitFlip","LottoCoin","BashCoin","DigiMoney","Lizus Payment","Crypto Improvement Fund","Stipend","Poa Network","Pushi","Ellerium","Velox","Ontology","Callisto Network","BitTube","Poseidon","Manna","Aidos Kuneen","Seraph","Cosmo","Bitspace","Briacoin","Ignition","Bitrolium","MedicCoin","Alpenschillling","Bitcoin Green","Deviant Coin","Abjcoin","Rapture","Semux","FuturoCoin","Carebit","Zealium","Monero Classic","Proton","iDealCash","Jumpcoin","Infinex","Bitcoin Incognito","KEYCO","HollyWoodCoin","GINcoin","Parlay","Listerclassic Coin","PlatinCoin","BetKings","Cognitio","Loki","Newton Coin","Mercoin","Swisscoin","Reliance","Xt3ch","MassGrid","TheVig","PluraCoin","EmaratCoin","Dekado","Lynx","Poseidon Quark","MYCE","Arionum","BitcoinWSpectrum","Muse","Motion","PlusOneCoin","Axe","GambleCoin","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","HexCoin","Deimos","TPCash","Webchain","Ryo","Urals Coin","Qwertycoin","Bitcoin Nova","ARENON","DACash","EUNO","MMOCoin","Ketan","Project Pai","XDNA","PAXEX","Azart","Averopay","ThunderStake","SimpleBank","Kcash","Xchange","Acute Angle Cloud","CrypticCoin","Bettex coin","Brazio","Actinium","TWIST","Bitcoin SV","DACH Coin","Argoneum","BitMoney","Junson Ming Chan Coin","FREDEnergy","HerbCoin","BlakeBitcoin","Universal Molecule","Lithium","Electron","PirateCash","Exosis","Block-Logic","Oduwa","Beam","Galilel","Bithereum","Crypto Sports","Credit","Scribe Network","SLICE","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","BLAST","Bitcoin Rhodium","GlobalToken","Insane Coin","ALAX","Media Protocol Token","LiteDoge","SolarCoin","TruckCoin","UFO Coin","OrangeCoin","BlakeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","Crypto Escudo","AudioCoin","IncaKoin","Piggy Coin","Crown Coin","Genstake","SmileyCoin","XiaoMiCoin","Groestlcoin","CapriCoin"," ClubCoin","Radium","Bata","Pakcoin","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","Fiii","JoinCoin","Triangles Coin","Vollar","TecraCoin","Gexan","EOS","Reality Clash","Oxycoin","TigerCash","LAPO","Waves","Particl","ShardCoin","BitTorrent","ChainZilla","Nxt","ZEPHYR","Gapcoin","Beldex","Horizen","BitcoinPlus","DivotyCoin"],"ProofType":["PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoC","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW","PoS","PoS/PoW/PoT","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoST","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoC","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/nPoS","PoW","PoW","PoW","PoW/PoS","PoW","PoS/PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoC","PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","dPoW/PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","DPoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","TPoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS ","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","Proof of Authority","PoW","PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","DPoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoS","PoW","PoW","PoW","PoS","PoS","PoW","PoW and PoS","PoW","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoW","PoS","POBh","PoW + Hive","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoA","PoW/PoS","PoW/PoS","PoS","PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoS","PoW/PoS","PoW","DPoS","PoW","PoW/PoS","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoS/PoW","PoW","PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","HPoW","PoS","PoW/PoS","PoW","PoS","PoS","PoW","PoW","PoW","PoS","PoW/PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","Zero-Knowledge Proof","PoW","DPOS","PoW","PoS","PoW","PoW","PoS","PoW","PoS","PoW","Pos","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","DPoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","Proof of Trust","PoW/PoS","DPoS","PoS","PoW/PoS","PoW/PoS","DPoC","PoW","PoW/PoS","PoW","PoW","PoW/PoS","DPoS","PoW","DPoS","PoS","PoW/PoS","LPoS","PoS","PoS","DPoS","DPoW","PoS/LPoS","DPoS","PoW/PoS","PoW","PoW","PoS","PoW/PoS"],"TotalCoinSupply":["42","532000000","314159265359","21000000","0","84000000","22000000","0","210000000","21000000","3600570502","21000000000","22000000","12500000","21000000","350000000","20000000","64000000","16768584","0","2000000000","384000000000","169795588","250000000","100000000","0","247000000","84000000","48166000","500000","21000000 ","1000000","13140000","18000000","40000000000","4380000","16555000000","21000000000","13500000000","21000000 ","299792458","32000000","0","0","11235813","336000000","1420609614","70000000","72245700","120000000","0","90600000000","22000000","21000000","265420800","20000000","1350000","100000000","42000000","200000000","0","10000000","2419200000","16180000","0","15000000000","21000000","337000000","0","3770000","420000000","98000000","0","21000000","250000000","51200000","0","1000","888000000","100000000","47011968","2300000000","10000000","42000000","80000000","250000","0","1600000000","100000000","23000000","0","84000000","265420800","5500000","0","45000000","0","1000000000","184467440735","10000000000","2158812800","500000000","105120000","68000000","0","1680000000","0","166386000","12000000","2628000","500000000","160000000","0","10000000","1000000000","1000000000","20000000","0","0","3371337","20000000","10000000000","1840000000","619478","21000000","92000000000","0","78000000","33000000","65789100","53760000","5060000","21000000","0","210240000","250000000","100000000","16906397","50000000","0","1000000000","26298000","16000000","21000000","21000000","210000000","81962100","22000000","26550000000","84000000","10500000","21626280000 ","0","42000000","221052632","84000000","30000000","168351300","24000000","384000000"," 99000000000","40000000","2147483647","20000000","20000000","84000000","25000000","75000000","222725000","525000000","90000000","139000000","616448","33500000","2000000000","44333333","100000000","200000000","25000000","657000000","200000000","50000000","90000000","10000000","21000000","9736000","3000000","21000000","1200000000","0","200000000","0","10638298","3100000000","30000000","20000000000","74000000","0","1500000000","21400000","39999898","2500124","100000000","301000000","7506000000","26205539","21000000","125000000","30000000","10000000000","500000000","850000000","3853326.77707314","50000000","38540000 ","42000000","228000000","20000000","60000000","20000000","33000000","76500000","28000000","650659833","5000000","21000000","144000000","32514916898","13000000","3315789","15000000","78835200","2714286","25000000","9999999","500000000","21000000","9354000","20000000","100000000000","21933333","185000000","55000000","110000000","3360000","14524851.4827","1000000000","17000000","1000000000","100000000 ","21000000","34426423","2232901","100000000","36900000","110000000","4000000000","110290030","100000000","48252000","400000000","500000000","21212121","28600000","1000000000","75000000000","40000000","35520400","2000000000","2500000000","30000000","105000000","90000000000","200084200","72000000","100000000","105000000","50000000000","0","340282367","30000000","2000000000","10000000","100000000","120000000","100000000","19276800","30000000"," 75000000","60000000","18900000","50000000000","54000000","18898187.6216583","21000000","200000000000","5000000000","137500000","1100000000","100000000","21000000","9507271","17405891.19707116","86712634466","10500000000","61599965","0","20000000","84000000","100000000","100000000","48252000","4200000","88888888","91388946","45000000000","27000000","21000000000","400000000","1800000000","15733333","27000000","24000000","21000000","25000000000","100000000","1200000000","60000000","156306732.71","21000000","200000000","100000000","21000000","14788275.991","9000000000","350000000","45000000","280000000","31800000","57879300","144000000","500000000","30000000000","210000000","660000000","210000000","85000000000","12500000","10000000000","27716121","100000000","20000000","550000000","999481516","50000000000","150000000","4999999999","50000000"," 90000000000","19800000","50000000","70000000","21000000","32000000","120000000","500000000","64000000","900000000","4000000","21000000","25000000000","23000000","20166000","23000000","25000000","1618033","24000000","30000000","1000000000000","16600000","232000000","336000000","3980000000","10000000000","25000000","100000000","1100000000","800000000","5200000000","840000000","40000000","18406979840","72000000","25000000","69000000","500000000","19340594","252460800","25000000","60000000","124000000","1000000000","6500000000","1000000000","21000000","10044655075.56243680","25000000","32000000","11892000","50000000","3000000","5000000","70000000","500000000","300000000","21000000","88000000","30000000","21000000","100000000","100000000","200000000","80000000","18400000","45000000","5121951220","21000000","26280000","21000000","18000000","26000000","10500000","30000000","110000000","600000518","8148139","7500000","150000000","184000000000","100000000","10200000000","62000000","44000000","168000000","100000000","1000000000","84000000","90000000","92000000000","650000000 ","250000000","545399000","100262205","18081806 ","22075700","21000000","21000000","15600000","82546564","21000000","5151000","16880000000","52500000","100000000","22105263","1000000000","1000000000","1750000000","88188888","210000000","184470000000","1000000000","55000000","50000000","50000000","260000000","210000000","2100000000","366000000","100000000","25000000","36500000","18000000000","21000000","1000000000","100000000","1000000000","7600000000","50000000","207000000","84000000","200000000","21000000","38000000","64000000","70000000000","0","8080000000","54000000","21000000","105120001.44","25228800","7000000000","105000000","21000000","120000000","21000000","262800000","19035999","30886000","13370000","74800000000","32700000","100000000","19700000","84000000","500000000","5000000000","420000000","64000000","2100000","168000000","30000000","1000000000","10000000000","35000000000","98100000000","0","4000000000","200000000","7000000000","54256119","21000000","0","500000000","1000000000","10500000000","190000000","1000000000","42000000","15000000","50000000000","400000000","105000000","208000000","160000000","9000000","5000000","182000000","16504333","105000000","159918400","10008835635","300000000","60168145","5000000000","2800000","120000","2100000000","210000000","21000000","0","24487944","0","1000000000","100000000000","100000000","8634140","900000000","990000000000","11000000","1000000000","2000000000","250000000","1400222610","21000000","1000000","100000000"],"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[578]}},"selected":{"id":"1206"},"selection_policy":{"id":"1240"}},"id":"1205","type":"ColumnDataSource"},{"attributes":{"editor":{"id":"1208"},"field":"CoinName","formatter":{"id":"1207"},"title":"CoinName"},"id":"1209","type":"TableColumn"}],"root_ids":["1203"]},"title":"Bokeh Application","version":"2.1.1"}};
  var render_items = [{"docid":"fa1300db-27a8-4419-b6e1-bcb4bf921d60","root_ids":["1203"],"roots":{"1203":"3599fbd4-cfa4-474a-bee3-90708add65bf"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python
# Print the total number of tradable cryptocurrencies
len(clustered_df)
```




    578




```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

#### Scatter Plot with Tradable Cryptocurrencies


```python
# Scale data to create the scatter plot

scaled_data = scaler.fit_transform(clustered_df[["TotalCoinsMined", "TotalCoinSupply"]])
```


```python
scaled = pd.DataFrame(scaled_data, columns= ["TotalCoinsMined", "TotalCoinSupply"], index =crypto_df2.index)
scaled = pd.concat([scaled, coin_name], axis = 1)
scaled["Class"] = model.labels_
```


```python
# Plot the scatter with x="TotalCoinsMined" and y="TotalCoinSupply"

scaled.hvplot.scatter(x="TotalCoinsMined", y="TotalCoinSupply", hover_cols=["CoinName"], by='Class')
```




<div id='1360'>





  <div class="bk-root" id="725d06bf-5015-44e4-b748-a4717cb4437b" data-root-id="1360"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"15de1ec2-b9b0-47b7-80ca-dbb4a4eb8593":{"roots":{"references":[{"attributes":{},"id":"1385","type":"SaveTool"},{"attributes":{"data":{"Class":[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],"CoinName":["42 Coin","404Coin","EliteCoin","Dash","Bitshares","BitcoinDark","PayCoin","KoboCoin","Aurora Coin","BlueCoin","EnergyCoin","BitBar","CryptoBullion","CasinoCoin","Diamond","Exclusive Coin","FlutterCoin","HoboNickels","HyperStake","IOCoin","MaxCoin","MintCoin","MazaCoin","Nautilus Coin","NavCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","SuperCoin","SyncCoin","TeslaCoin","TittieCoin","TorCoin","UnitaryStatus Dollar","UltraCoin","VeriCoin","X11 Coin","Crypti","StealthCoin","ZCC Coin","BurstCoin","StorjCoin","Neutron","FairCoin","RubyCoin","Kore","Dnotes","8BIT Coin","Sativa Coin","Ucoin","Vtorrent","IslaCoin","Nexus","Droidz","Squall Coin","Diggits","Paycon","Emercoin","EverGreenCoin","Decred","EDRCoin","Hitcoin","DubaiCoin","PWR Coin","BillaryCoin","GPU Coin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","Lutetium Coin","GoldBlocks","CarterCoin","BitTokens","MustangCoin","ZoneCoin","RootCoin","BitCurrency","Swiscoin","BuzzCoin","Opair","PesoBit","Halloween Coin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","ARK","InsaneCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","Impeach","Zilbercoin","FirstCoin","FindCoin","OpenChat","RenosCoin","VirtacoinPlus","TajCoin","Impact","Atmos","HappyCoin","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","GrowthCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","QTUM","Espers","Denarius","Virta Unique Coin","Bitcoin Planet","BritCoin","Linda","DeepOnion","Signatum","Cream","Monoeci","Draftcoin","Stakecoin","CoinonatX","Ethereum Dark","Obsidian","Cardano","Regalcoin","TrezarCoin","TerraNovaCoin","Rupee","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Xios","Bitcloud 2.0","KekCoin","BlackholeCoin","Infinity Economics","Alqo","Magnet","Lamden Tau","Electra","Bitcoin Diamond","Cash & Back Coin","Bulwark","Kalkulus","GermanCoin","LiteCoin Ultra","PhantomX","Accolade","OmiseGO Classic","Digiwage","Trollcoin","Litecoin Plus","Monkey Project","ECC","TokenPay","My Big Coin","Unified Society USDEX","BitSoar Coin","Credence Coin","Tokyo Coin","BiblePay","BashCoin","DigiMoney","Lizus Payment","Stipend","Pushi","Ellerium","Velox","Ontology","Seraph","Bitspace","Briacoin","Ignition","MedicCoin","Bitcoin Green","Deviant Coin","Abjcoin","Semux","Carebit","Zealium","Proton","iDealCash","Bitcoin Incognito","HollyWoodCoin","Parlay","Listerclassic Coin","BetKings","Cognitio","Mercoin","Swisscoin","Reliance","Xt3ch","TheVig","EmaratCoin","Dekado","Lynx","Poseidon Quark","MYCE","BitcoinWSpectrum","Muse","GambleCoin","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","TPCash","ARENON","EUNO","MMOCoin","Ketan","XDNA","PAXEX","Averopay","ThunderStake","SimpleBank","Kcash","Bettex coin","TWIST","DACH Coin","BitMoney","Junson Ming Chan Coin","HerbCoin","Oduwa","Galilel","Crypto Sports","Credit","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","Insane Coin","ALAX","LiteDoge","TruckCoin","OrangeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","AudioCoin","IncaKoin","Piggy Coin","Genstake","XiaoMiCoin","CapriCoin"," ClubCoin","Radium","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","Fiii","Triangles Coin","Gexan","EOS","Oxycoin","TigerCash","LAPO","Particl","ShardCoin","Nxt","ZEPHYR","Gapcoin","BitcoinPlus","DivotyCoin"],"TotalCoinSupply":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[313]},"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[313]}},"selected":{"id":"1461"},"selection_policy":{"id":"1480"}},"id":"1460","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#2ba02b"},"line_color":{"value":"#2ba02b"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1442","type":"Scatter"},{"attributes":{},"id":"1382","type":"BasicTicker"},{"attributes":{},"id":"1400","type":"BasicTickFormatter"},{"attributes":{},"id":"1480","type":"UnionRenderers"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"1445"}]},"id":"1459","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7e0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#ff7e0e"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1424","type":"Scatter"},{"attributes":{},"id":"1416","type":"UnionRenderers"},{"attributes":{"axis_label":"TotalCoinsMined","bounds":"auto","formatter":{"id":"1398"},"major_label_orientation":"horizontal","ticker":{"id":"1378"}},"id":"1377","type":"LinearAxis"},{"attributes":{},"id":"1457","type":"UnionRenderers"},{"attributes":{"axis_label":"TotalCoinSupply","bounds":"auto","formatter":{"id":"1400"},"major_label_orientation":"horizontal","ticker":{"id":"1382"}},"id":"1381","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1364"},{"id":"1385"},{"id":"1386"},{"id":"1387"},{"id":"1388"},{"id":"1389"}]},"id":"1391","type":"Toolbar"},{"attributes":{},"id":"1389","type":"ResetTool"},{"attributes":{},"id":"1436","type":"UnionRenderers"},{"attributes":{},"id":"1373","type":"LinearScale"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"1408"}]},"id":"1419","type":"LegendItem"},{"attributes":{"callback":null,"renderers":[{"id":"1408"},{"id":"1426"},{"id":"1445"},{"id":"1466"}],"tags":["hv_created"],"tooltips":[["Class","@{Class}"],["TotalCoinsMined","@{TotalCoinsMined}"],["TotalCoinSupply","@{TotalCoinSupply}"],["CoinName","@{CoinName}"]]},"id":"1364","type":"HoverTool"},{"attributes":{},"id":"1378","type":"BasicTicker"},{"attributes":{"end":16.749657454799973,"reset_end":16.749657454799973,"reset_start":-1.6871251944219514,"start":-1.6871251944219514,"tags":[[["TotalCoinSupply","TotalCoinSupply",null]]]},"id":"1363","type":"Range1d"},{"attributes":{"text":"","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1369","type":"Title"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02242","sizing_mode":"stretch_width"},"id":"1361","type":"Spacer"},{"attributes":{"axis":{"id":"1381"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1384","type":"Grid"},{"attributes":{"source":{"id":"1402"}},"id":"1409","type":"CDSView"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"1466"}]},"id":"1482","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ba02b"},"line_alpha":{"value":0.1},"line_color":{"value":"#2ba02b"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1443","type":"Scatter"},{"attributes":{},"id":"1440","type":"Selection"},{"attributes":{"data_source":{"id":"1402"},"glyph":{"id":"1405"},"hover_glyph":null,"muted_glyph":{"id":"1407"},"nonselection_glyph":{"id":"1406"},"selection_glyph":null,"view":{"id":"1409"}},"id":"1408","type":"GlyphRenderer"},{"attributes":{},"id":"1398","type":"BasicTickFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02243","sizing_mode":"stretch_width"},"id":"1609","type":"Spacer"},{"attributes":{"source":{"id":"1460"}},"id":"1467","type":"CDSView"},{"attributes":{"data_source":{"id":"1420"},"glyph":{"id":"1423"},"hover_glyph":null,"muted_glyph":{"id":"1425"},"nonselection_glyph":{"id":"1424"},"selection_glyph":null,"view":{"id":"1427"}},"id":"1426","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"1390"}},"id":"1388","type":"BoxZoomTool"},{"attributes":{},"id":"1461","type":"Selection"},{"attributes":{"end":23.472041566476683,"reset_end":23.472041566476683,"reset_start":-1.2241791587421138,"start":-1.2241791587421138,"tags":[[["TotalCoinsMined","TotalCoinsMined",null]]]},"id":"1362","type":"Range1d"},{"attributes":{"children":[{"id":"1361"},{"id":"1368"},{"id":"1609"}],"margin":[0,0,0,0],"name":"Row02238","tags":["embedded"]},"id":"1360","type":"Row"},{"attributes":{"data":{"Class":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"CoinName":["Bitcoin","Ethereum","Litecoin","Monero","Ethereum Classic","ZCash","DigiByte","ProsperCoin","Spreadcoin","Argentum","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","QuarkCoin","Riecoin","Digitalcoin ","Catcoin","CannaCoin","CryptCoin","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","Infinite Coin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MegaCoin","MediterraneanCoin","MinCoin","NobleCoin","Namecoin","NyanCoin","RonPaulCoin","StableCoin","SmartCoin","SysCoin","TigerCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","ViaCoin","Vertcoin","WorldCoin","JouleCoin","ByteCoin","DigitalNote ","MonaCoin","Gulden","PesetaCoin","Wild Beast Coin","Flo","ArtByte","Folding Coin","Unitus","CypherPunkCoin","OmniCron","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","MarsCoin ","Crypto","Anarchists Prime","BowsCoin","Song Coin","BitZeny","Expanse","Siacoin","MindCoin","I0coin","Revolution VR","HOdlcoin","Gamecredits","CarpeDiemCoin","Adzcoin","SoilCoin","YoCoin","SibCoin","Francs","Aiden","BolivarCoin","Omni","PizzaCoin","Komodo","Karbo","ZayedCoin","Circuits of Value","DopeCoin","DollarCoin","Shilling","ZCoin","Elementrem","ZClassic","KiloCoin","ArtexCoin","Kurrent","Cannabis Industry Coin","OsmiumCoin","Bikercoins","HexxCoin","PacCoin","Citadel","BeaverCoin","VaultCoin","Zero","Canada eCoin","Zoin","DubaiCoin","EB3coin","Coinonat","BenjiRolls","ILCoin","EquiTrader","Quantum Resistant Ledger","Dynamic","Nano","ChanCoin","Dinastycoin","DigitalPrice","Unify","SocialCoin","ArcticCoin","DAS","LeviarCoin","Bitcore","gCn Coin","SmartCash","Onix","Bitcoin Cash","Sojourn Coin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","Sumokoin","BitcoinZ","Elements","VIVO Coin","Bitcoin Gold","Pirl","eBoost","Pura","Innova","Ellaism","GoByte","SHIELD","UltraNote","BitCoal","DaxxCoin","BoxyCoin","AC3","Lethean","PopularCoin","Photon","Sucre","SparksPay","GoaCoin","GunCoin","IrishCoin","Pioneer Coin","UnitedBitcoin","Interzone","1717 Masonic Commemorative Token","Crypto Wisdom Coin","TurtleCoin","MUNcoin","Niobio Cash","ShareChain","Travelflex","KREDS","BitFlip","LottoCoin","Crypto Improvement Fund","Callisto Network","BitTube","Poseidon","Manna","Aidos Kuneen","Cosmo","Bitrolium","Alpenschillling","Rapture","FuturoCoin","Monero Classic","Jumpcoin","Infinex","KEYCO","GINcoin","PlatinCoin","Loki","Newton Coin","MassGrid","PluraCoin","Arionum","Motion","PlusOneCoin","Axe","HexCoin","Deimos","Webchain","Ryo","Urals Coin","Qwertycoin","Bitcoin Nova","DACash","Project Pai","Azart","Xchange","CrypticCoin","Brazio","Actinium","Bitcoin SV","Argoneum","FREDEnergy","BlakeBitcoin","Universal Molecule","Lithium","Electron","PirateCash","Exosis","Block-Logic","Beam","Bithereum","Scribe Network","SLICE","BLAST","Bitcoin Rhodium","GlobalToken","Media Protocol Token","SolarCoin","UFO Coin","BlakeCoin","Crypto Escudo","Crown Coin","SmileyCoin","Groestlcoin","Bata","Pakcoin","JoinCoin","Vollar","TecraCoin","Reality Clash","ChainZilla","Beldex","Horizen"],"TotalCoinSupply":{"__ndarray__":"m+dXK3BAw78wy2ezAkvDv948KJO4IMO/MMtnswJLw79j58hiSeHCv5vnVytwQMO/CTGpymABxj+b51crcEDDv8dHQA3xQMO/R8BPOcoqw79t7PCXO7i+v36/AFcF/xZAWLNxB4f1wr+R4HkWJs3CvyM6onSqGMO/FAEzvKjOwr/ePCiTuCDDv82o2PLCMsO/m+dXK3BAw7/oQXUvZUTDvx4IEdHyQcO/cba2lRyHuj8JManKYAHGP4rJHpsZBq0/m+dXK3BAw7/sOLa2FLTCv7vFW3bmOsO/kjcHmlpFw7/nkWky2qHCv3wL5gzPf8C/QX/d7cQnw79vWHJ/oybDv15xeW4p3PM/m+dXK3BAw7/UIQaeYsXCv8dHQA3xQMO/W0mhtVRKw78HBEij3TXDvxep3DVS5sK/ewlU4PlFw787ZLYoZGm0P5vnVytwQMO/vDGBUFmhwr+b51crcEDDv5HgeRYmzcK/Qo9U6zsxw7/LM1oV84vBv5Qj365XM8O/BwRIo901w7+MvckavCLDvyWz7XriSsO/MMtnswJLw79EJ4dnbj/Dv948KJO4IMO/1CEGnmLFwr+E4479WjTDv0JBEVapdwVAkWs3cXTdZz/ygNOTFhbDv4pZ4Vxw+r+/uXITdz73wr+IfGr/r0nDv+mvK4J1+sK/syCwP5BTwb+zILA/kFPBvzDLZ7MCS8O/MMtnswJLw79B1tIxUEnDv5FrN3F03Wc//CJp+lVZv797zWfcskrDv5vnVytwQMO/NC9T3EM09D+QZXOUZTrDv2VG6aPjKcO/qTLt+vEvw7+b51crcEDDv7Q2P3Qq4cK/keB5FibNwr/ckE3Df0LDvzDLZ7MCS8O/dsjhlPRCw7+b51crcEDDv2PnyGJJ4cK/TxRxOb8hw7/ePCiTuCDDv+OpL0utPMc/3jwok7ggw78Thiw66DvDv4ctJixB9sK/GMeehe0+w7/HR0AN8UDDv948KJO4IMO/7Wa2o2w+w79ahGBAs0rDv+1mtqNsPsO/F6ncNVLmwr97CVTg+UXDv9VLuOYbRsO/mv4kwt/uwL8Xqdw1UubCv8MJYpynRcO/E4YsOug7w78kwcedPEDDvzYeRUTRPcO/m+dXK3BAw7+RazdxdN1nP/L1i3lJT8K/USRygDnYwr+b51crcEDDv6TugOCkScO/7Wa2o2w+w7/Fe1zg+UXDv7DZClC2K/Y/pkt6ct/twr+nIeCnUUnDv7MgsD+QU8G/Smj5snNCw78jOqJ0qhjDv5vnVytwQMO/56FG6+JJw797QhLJcdq2v33JXt23MsO/6xKbviA5w7/xQTkkycC8v+m+DCrDJsO/SVkYCyYWw78wy2ezAkvDv6gJZEayn8K/E4YsOug7w79t7PCXO7i+vyM6onSqGMO/x/VxQk5Bw79nnlOEQCXDv/VA8cDNLMO/kbGM0n5Bw7/7gWMM0y/Dv5vnVytwQMO/Y1ZBe2ZgB0CC7aLhjOuyv6eP6gA4IcG/m+dXK3BAw7/JLgu68LKFPzDLZ7MCS8O/x0dADfFAw7/ePCiTuCDDvyM6onSqGMO/IzqidKoYw78RPqN7Qh7DvwkxqcpgAcY/nzAHk5yBv7+VpuXfaj3Dv5vnVytwQMO/v6R9gVH8wr8jOqJ0qhjDv4RPtNfNmsK/hOOO/Vo0w79zmz6dC77Cv/fYIz0AO8O/qtpPSLz+wb8HehK3v3vyPw4Zj6u3RMO/kWs3cXTdZz8jOqJ0qhjDv2stKVodNsK/yVGAEtNTwb8V0rPhjOuyv5REZT9ntvM/A1sI1ApBw7+b51crcEDDv7vFW3bmOsO/8vWLeUlPwr9HwE85yirDv0Qnh2duP8O/ky1DqNtAw79EJ4dnbj/Dv4mILCoySsO/GMeehe0+w7+uKEhFMG0uQMKOiUCnQsO/55FpMtqhwr+RazdxdN1nPyM6onSqGMO/p4/qADghwb9exBhn3zbDvwvayv7t58A/8vWLeUlPwr8Y3PcMawqqv7MgsD+QU8G/m+dXK3BAw7+Ly4TNRHxtP+1mtqNsPsO/ptjaBwZFw79Bf93txCfDvwsYF/f5s8K/m+dXK3BAw78jOqJ0qhjDv6bhgEO/QcO/m+dXK3BAw7+guIKrxz3Dvx4IEdHyQcO/Ztlfb7lFw79UI68p8RzCv51xP1V+/8K/5quJB/RoBUCMruhybvbCv7MgsD+QU8G/ilghV244wr8c/suH5T/Dv5vnVytwQMO/m+dXK3BAw78z4GW44T/Dv7MgsD+QU8G/rMHM0fSzv79jgV+znB7Dv2PnyGJJ4cK/6LST8713BUCzILA/kFPBv6oCBZTWMcO/VMplGotTvr/tZrajbD7DvyM6onSqGMO/8+0CQ0Bjob/lB4IIzOLCv948KJO4IMO/m+dXK3BAw79HwE85yirDvztNY+hDOZu/m+dXK3BAw79AV8eTFhbDvxfYtCZPPsO/HoeIJYYbpr9JWRgLJhbDv5vnVytwQMO/u7Z6zpgOw7+VEXVktMbCv9NZkgl2O8O/aoKfPow6w78jOqJ0qhjDv0fATznKKsO/OwEzDPRJw7+MruhybvbCv5FrN3F03Wc/M6H1uiS09T97QhLJcdq2vx6HiCWGG6a/syCwP5BTwb8HBEij3TXDv0rgnflVwuM/SVkYCyYWw79W6t1JfkjDvylsMxhi78K/6b121JlJw79UymUai1O+v2PnyGJJ4cK/A+Bwoq4+w79QqWv+eEXDv/LE05MSisC/m+dXK3BAw78=","dtype":"float64","order":"little","shape":[259]},"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[259]}},"selected":{"id":"1403"},"selection_policy":{"id":"1416"}},"id":"1402","type":"ColumnDataSource"},{"attributes":{"source":{"id":"1439"}},"id":"1446","type":"CDSView"},{"attributes":{"align":null,"below":[{"id":"1377"}],"center":[{"id":"1380"},{"id":"1384"}],"left":[{"id":"1381"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1408"},{"id":"1426"},{"id":"1445"},{"id":"1466"}],"right":[{"id":"1418"}],"sizing_mode":"fixed","title":{"id":"1369"},"toolbar":{"id":"1391"},"x_range":{"id":"1362"},"x_scale":{"id":"1373"},"y_range":{"id":"1363"},"y_scale":{"id":"1375"}},"id":"1368","subtype":"Figure","type":"Plot"},{"attributes":{"data_source":{"id":"1439"},"glyph":{"id":"1442"},"hover_glyph":null,"muted_glyph":{"id":"1444"},"nonselection_glyph":{"id":"1443"},"selection_glyph":null,"view":{"id":"1446"}},"id":"1445","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#1f77b3"},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1405","type":"Scatter"},{"attributes":{},"id":"1421","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1390","type":"BoxAnnotation"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#ff7e0e"},"line_alpha":{"value":0.2},"line_color":{"value":"#ff7e0e"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1425","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62628"},"line_alpha":{"value":0.1},"line_color":{"value":"#d62628"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1464","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1406","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#d62628"},"line_alpha":{"value":0.2},"line_color":{"value":"#d62628"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1465","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.2},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1407","type":"Scatter"},{"attributes":{"data":{"Class":[2],"CoinName":["BitTorrent"],"TotalCoinSupply":{"__ndarray__":"C3YzY4YeLkA=","dtype":"float64","order":"little","shape":[1]},"TotalCoinsMined":{"__ndarray__":"ACaU5Ed/NkA=","dtype":"float64","order":"little","shape":[1]}},"selected":{"id":"1440"},"selection_policy":{"id":"1457"}},"id":"1439","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#d62628"},"line_color":{"value":"#d62628"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1463","type":"Scatter"},{"attributes":{"source":{"id":"1420"}},"id":"1427","type":"CDSView"},{"attributes":{"data_source":{"id":"1460"},"glyph":{"id":"1463"},"hover_glyph":null,"muted_glyph":{"id":"1465"},"nonselection_glyph":{"id":"1464"},"selection_glyph":null,"view":{"id":"1467"}},"id":"1466","type":"GlyphRenderer"},{"attributes":{},"id":"1375","type":"LinearScale"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"1426"}]},"id":"1438","type":"LegendItem"},{"attributes":{"fill_color":{"value":"#ff7e0e"},"line_color":{"value":"#ff7e0e"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1423","type":"Scatter"},{"attributes":{},"id":"1403","type":"Selection"},{"attributes":{"click_policy":"mute","items":[{"id":"1419"},{"id":"1438"},{"id":"1459"},{"id":"1482"}],"location":[0,0],"title":"Class"},"id":"1418","type":"Legend"},{"attributes":{"axis":{"id":"1377"},"grid_line_color":null,"ticker":null},"id":"1380","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#2ba02b"},"line_alpha":{"value":0.2},"line_color":{"value":"#2ba02b"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1444","type":"Scatter"},{"attributes":{},"id":"1387","type":"WheelZoomTool"},{"attributes":{},"id":"1386","type":"PanTool"},{"attributes":{"data":{"Class":[1,1,1,1,1],"CoinName":["Vechain","LitecoinCash","Poa Network","Acute Angle Cloud","Waves"],"TotalCoinSupply":{"__ndarray__":"LEYDyIbn8j/6O+xwHaTBv5lmEe/oy8K/syCwP5BTwb8jOqJ0qhjDvw==","dtype":"float64","order":"little","shape":[5]},"TotalCoinsMined":{"__ndarray__":"UkatPVVw8j8rGc4rJH65v/iqYlmgB7y/doRBQNtgt791hNfbA6S8vw==","dtype":"float64","order":"little","shape":[5]}},"selected":{"id":"1421"},"selection_policy":{"id":"1436"}},"id":"1420","type":"ColumnDataSource"}],"root_ids":["1360"]},"title":"Bokeh Application","version":"2.1.1"}};
  var render_items = [{"docid":"15de1ec2-b9b0-47b7-80ca-dbb4a4eb8593","root_ids":["1360"],"roots":{"1360":"725d06bf-5015-44e4-b748-a4717cb4437b"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python

```
