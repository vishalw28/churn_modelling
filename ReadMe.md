#1. Create a new virtual env i.e. venv
```
conda create -p venv python==3.11 -y
```

#2. Add all the required libraries into `requirements.txt` file

#3. Activate environment
```
conda activate venv/
```
then 
```
pip install -r requirements.txt
```

#4. Load the data using pandas lib

#5. Drop the irrelevant columns

#6. LabelEncoder to covert data which contains only 2 values
e.g. Gender i.e. Male & Female
    HasCreditCard i.e. Yes/No
    Basically LabelEncoder coverts this data to 1 & 0

#7. What about data which has multiple values (LabelEncoder valeus)    
In that case LabelEncoder gives values 0,1,2,3 etc.
Hence `OneHotEncoder` helps

-> Convert the Geography using OHE
```
from sklearn.preprocessing import OneHotEncoder
ohe_geo=OneHotEncoder()
geo_encoder=ohe_geo.fit_transform(data[['Geography']])
geo_encoder

ohe_geo.get_feature_names_out(['Geography']) # Gives OHE matrix
```

-> Next concat the OHE output to main data

```
geo_encoded_df=pd.DataFrame(geo_encoder.toarray(), columns=ohe_geo.get_feature_names_out(['Geography']))
geo_encoded_df

data=pd.concat([data.drop('Geography',axis=1,errors='ignore'), geo_encoded_df],axis=1)
data.head()
```

#8. Save the encoders & scaler into pickle file
```
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('ohe_geo.pkl','wb') as file:
    pickle.dump(ohe_geo, file)
```

#9. Also write the scaler to to picker file
    
```
## a. Divide the dataset into dependent v/s independent feature
x=data.drop('Exited', axis=1)
y=data['Exited']

## Split the data in training & testing sets
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


## b. Scale these features
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
```
Next write the data to the file
```
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)
```

#10 Now your all the data is ready for AI.

Next ANN implementation