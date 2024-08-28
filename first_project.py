from os import system
import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit , cross_val_score , GridSearchCV , RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer , IterativeImputer
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder , StandardScaler , MinMaxScaler , FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import  TransformedTargetRegressor , ColumnTransformer , make_column_selector , make_column_transformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator , TransformerMixin 
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy.stats import randint , t ,sem
#________________________________________________________-get_data______________________________________________________-
def load_data() -> pd.DataFrame:
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.exists():
        print('in if')
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)  # Use urlretrieve for downloading
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')

    csv_path = Path('datasets/housing/housing.csv')
    return pd.read_csv(csv_path)

housing = load_data()

(
    #________________________________________________________-some_processing_____________________________________________-
#print(len(housing))
#print(housing.head())
#print(housing.info())
#print(housing['ocean_proximity'].value_counts() / len(housing['ocean_proximity'] )) # return values num (how many 'ex' in data) like : {index : <1H OCEAN     his count :9136}
#print(housing['ocean_proximity'].sort_values(ascending=False)) # will return sorted values from low to high ' ascending' fales from low to high else from high to low 
#print(housing['median_house_value'].value_counts())
#print(housing.describe())
#print(housing.isna().any()) # this func will return all columns if any columns has missing value will be write side it true else will write false
#print(housing.isna().sum()) # will return how many missing value in columns
#print(housing['median_income'].describe())
#null_rows = housing.isnull().any(axis=1) # get number and poistion missing value # to display it write housing.loc[null_rows]
#housing.hist(bins=50, figsize=(12,8)) # make data from texts to grapical
#plt.show() display grapical
)

#______________________________________________________________-split_data_to(train_set , test_set)________________________________________________________-
def split_data(data : pd.DataFrame , raito): #useless but will make you under stand how we split a data
    # this method can make the data is bias
    np.random.seed(90)
    shuffle_indicate = np.random.permutation(len(data))
    test_set_size = int(len(data) * raito)
    test_indices = shuffle_indicate[:test_set_size]
    train_indices =shuffle_indicate[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]

housing['income_cat'] = pd.cut(housing['median_income'] , bins = [0 , 1.5  ,3.0 , 4.5 , 6 , np.inf] , labels= [1,2,3,4,5]) # make category from medain_income in order to when you split data to train and test split it fair
#housing['house_income_cat']  = pd.cut(housing['median_house_value'] , bins = [0 , 50000 , 100000 , 240000 ,370000 , np.inf ] , labels= ['cheep' , 'mid' , ' avrage' , 'high' , 'expisev'])

def split_on_ground_using_stratifiedshufflesplit(data: pd.DataFrame , column , test_sizee , random_statee , n_splitss : int = 10 ) -> list[pd.DataFrame,pd.DataFrame]:
    # this func will split data too but with out bias using stratified...... this will split data on clear ground 'income_cat'
    spliter = StratifiedShuffleSplit(n_splits=n_splitss , test_size=test_sizee , random_state= random_statee) # we call this class and give it 1: number of group , 2: raito_test , 3: random seed
    training = []
    for train_index , test_index in spliter.split(data , column): # we here split data with data column to prevent bias with income_cat help
        start_train_set = housing.iloc[train_index] # we transform indexs to data and put it in train set 
        start_test_set = housing.iloc[test_index] # we transform indexs to data and put it in test set 
        training.append([start_train_set , start_test_set])
    return training

train_set , test_set = split_on_ground_using_stratifiedshufflesplit(housing , housing['income_cat'] , 0.2 , 42 , 10)[0]
#print(test_set['income_cat'].value_counts() / len(test_set))


for set_ in (train_set , test_set): # to del income_cat column
    #set_.drop('income_cat' , axis = 1 , inplace = True)
    # 'implace' mean any edit will be change the main data set and don't create new data frame 
    set_.drop('income_cat' , axis = 1 , inplace = True)

housing : pd.DataFrame = train_set.copy()
def display_some_data():
    '''
    kind : scatter  = each point equel  one value 
    x = put longtitude
    y = put latitude
    grid = grid
    s = this is radius of each circle - any column you will add it here he  will represent such as cicrle if value is high , radius of circle will be big , else will be small
    c = this is color of circles 
    cmap or color map = color gradation in map some values = [ocean , coolwarm , jet , viridis[default]]
    legend = if his value True it will put mark else: no
    '''
    housing.plot(kind = 'scatter' , x = 'longitude' , y = 'latitude' , grid = True , label = 'housing_age' , s = housing['housing_median_age'] , c = housing['median_house_value'],cmap = 'jet' , colorbar = True , legend= True , sharex= False , figsize=(10 , 7))


    #housing.plot(kind = 'scatter' , x = 'longitude' , y = 'latitude' , grid = True , label = 'population' , s = housing['population'] / 100, c = housing['median_house_value'] , cmap = 'jet' , colorbar = True , legend= False, sharex= False , figsize=(10 , 7))
    #housing.plot(kind = 'scatter' , x = 'longitude' , y = 'latitude' , grid = True , label = 'total_rooms' , s =  housing['total_rooms'] / 100, c = housing['housing_median_age'] , cmap = 'jet' , colorbar = True , legend= True , sharex= False , figsize=(10 , 7))
    plt.show()

#_____________________________________________________________________________________-ralate_between_each other data____________________-
def create_corr():
    matrix = housing.corr() # create range of relate between the coulmns
    print(matrix['median_income'].sort_values(ascending=False)) # will get all values of columns from 'median_income'
    matrix['median_house_value'].sort_values(ascending=False).plot(kind='line',figsize=(40 , 30)) #will draw line for all lines 
    plt.show()

# add some category to data set cuz we want more specifically and more useful
housing['rooms_per_house'] = housing['total_rooms'] / housing['households']
housing['bedrooms_raito'] = housing['total_bedrooms'] / housing['total_rooms'] # more important **
housing['people_per_house'] = housing['population'] / housing['households']



housing = train_set.drop('median_house_value' , axis = 1)
housing_labels = train_set['median_house_value'].copy()
null_rows = housing.isnull().any(axis=1)



#__________________________________________________________________ replace_any_missing_value_to_real_value_______________________________________

def replace_missing_values_in_data_using_simpleimputer(data : pd.DataFrame , type_new_value : str = 'median') -> pd.DataFrame:
    imputer = SimpleImputer(strategy=str(type_new_value.lower().strip())) # this class help you to replace missing value to real values by mean or median or custom
    data_num = data.select_dtypes(include=[np.number]) # this line just take a numbers values 
    return pd.DataFrame(imputer.fit_transform(data_num) , columns= data_num.columns , index=data_num.index) # this line will return to you new data frame with not missing value how?
    #i think all parmetars is clear but not first par let's go to  explain it 
    # imputer.fit.... this prompt will learn a model to replace missing value with else and will be teansform it to data after this pd.data... will return new data farme without missing values
 
housing_edited = replace_missing_values_in_data_using_simpleimputer(housing , 'median')

def  replace_missing_values_in_data_using_IterativeImputer(data : pd.DataFrame , iter : int = 10 ):
    #this is same concepts but this will prdict a miss value not put mean or madian etc...
    imputer = IterativeImputer(max_iter=iter , initial_strategy='median')
    data_num = data.select_dtypes(include=[np.number])
    return pd.DataFrame(imputer.fit_transform(data_num) , columns=data_num.columns , index= data_num.index)
#___________________________________________________________________transform_from_catgory_to_numerical_cat______________________________________
housing_cat = housing[['ocean_proximity']] # if we use only housing[...] will be return column and we need data  frame have one column not just one col  , to do this housing[[....]]

housing_num = housing.select_dtypes(include=[np.number]) # all columns have number values (witout catgories (ocean...column))

def  replace_category_to_nums_using_onehot_and_ordinal(data : pd.DataFrame , hot_or_ord : str = 'hot' , matrix_or_no : bool = True) -> np.ndarray :
    encoder = OneHotEncoder(sparse_output=matrix_or_no) , OrdinalEncoder() # here we use to most alg to convert from category to nums
    if hot_or_ord == 'hot': 
        return encoder[0].fit_transform(data) 
    elif hot_or_ord == 'ord':  # if you cat dont related use it like (bad , average , good , excllent)
        return encoder[1].fit_transform(data)

#housing_edited2.plot(kind='scatter' , legend= True , x='longitude' , y= 'latitude' , c = 'total_bedrooms' ,colorbar=True , colormap= 'jet' , )

housing_hot = replace_category_to_nums_using_onehot_and_ordinal(housing_cat)
#print(housing_hot.toarray()) # to convect from matrix to basic array
#____________________________________________________-data_scaler____________________________________________-
min_max_scaler = MinMaxScaler((-1 , 1))
housing_min__scaler = min_max_scaler.fit_transform(housing_num)
v = pd.DataFrame(housing_min__scaler , columns=housing_num.columns , index=housing_num.index)
std_scaler = StandardScaler(with_mean=False)
housing_std_scaler = std_scaler.fit_transform(housing_num)
p = pd.DataFrame((housing_std_scaler) , columns=housing_num.columns , index=housing_num.index)
def return_pediction_data_to_default():
    target_scaler = StandardScaler()
    scaled_l = target_scaler.fit_transform(housing_labels.to_frame()) # trans_data_
    model = LinearRegression()
    model.fit(housing[['median_income']] , scaled_l) # tech model on data
    some_data_new = housing[['median_income']].iloc[:5]
    scaled_prediction = model.predict(some_data_new) # prediction but his value like transformed data(stander scaler)
    prediction = target_scaler.inverse_transform(scaled_prediction) # real value 
    return prediction

def return_pediction_data_to_default_2():
    model_and_transform_data = TransformedTargetRegressor(LinearRegression() ,transformer=StandardScaler())
    model_and_transform_data.fit(housing[['median_income']] , housing_labels)
    some_data_new = housing[['median_income']].iloc[:5]
    prediction = model_and_transform_data.predict(some_data_new)
    return prediction

#print(return_pediction_data_to_default())
#print('$'*50)
#print(return_pediction_data_to_default_2())

log_transformer = FunctionTransformer(np.log , inverse_func= np.exp) 
log_pop = log_transformer.transform(housing[['population']])
baisc = pd.DataFrame(log_transformer.inverse_transform(log_pop) , columns=housing[['population']].columns , index=housing[['population']].index)
#--
rbf_transformar = FunctionTransformer(rbf_kernel , kw_args=(dict(Y =[[30]] , gamma = 0.1 )))
rb = rbf_transformar.transform(X=housing[['housing_median_age']])
#pd.DataFrame(rb , columns= housing[['housing_median_age']].columns , index=housing[['housing_median_age']].index).hist(bins=50 , figsize=(12,8))
#housing[['housing_median_age']].hist(bins=50 , figsize=(12,8))


class ClusterSimilary(BaseEstimator , TransformerMixin):
    def __init__(self , n_clusters = 10 , gamma = 1.0 , random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.clustrs_marks = None
        self.names = None
        self.labels = None
    def fit(self , X , Y = None , sample_weight=None):
        self.kmeans = KMeans(self.n_clusters , random_state=self.random_state)
        self.kmeans.fit(X , sample_weight=sample_weight)
        self.clustrs_marks = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_
        self.names = self.kmeans.fit_predict(X,sample_weight=sample_weight)
        return self
    def transform(self , X):
        return rbf_kernel(X , self.kmeans.cluster_centers_ , gamma=self.gamma)
    def get_feature_names_out(self , names = None):
        return [f'cluster{i}' for i in range(self.n_clusters)]
cluster_simil = ClusterSimilary(random_state=42)
simies = cluster_simil.fit_transform(housing[['latitude' , 'longitude']] , sample_weight = housing_labels)
'''
we will explain somethings 
cluster.marks = poistions of any mark of cluser
array[: , 0]
ex : list = np.array([1,2,3],[4,5,6])
<<< list[: , 0]
    1 , 4
this prompet return first index from ecah column
'''
#housing.plot(kind = 'scatter' , x = 'longitude' , y = 'latitude' , grid = True , label = 'housing_age' , s = housing['housing_median_age'] , c = housing['median_house_value'] ,cmap = 'jet' , colorbar = True , legend= True , sharex= False , figsize=(10 , 7))

def display_cluster_marks():

    plt.grid()
    #sns.kdeplot(housing['longitude'] , housing['latitude'] , cmap = 'Reds' ,shade = True , bw_adjust=0.5 , alpha = 0.6)
    plt.scatter(housing['longitude'] , housing['latitude'] , c = cluster_simil.names , cmap = 'rainbow' , s=50 , alpha=0.6 , edgecolors='w')
    plt.scatter(cluster_simil.clustrs_marks[:,1] ,cluster_simil.clustrs_marks[:,0] , c='black' , s=200  ,marker='X' , label = 'cluster_centers')
    plt.colorbar(label = 'more_simi')
    plt.show()
#system('cls')
#__________________________________________________________________________- piplines_________________________
num_attribs = [_ for _ in housing_num.columns] # all columns name  
cat_attribs = [_ for _ in housing_cat.columns] # all cat names (ocean..)

num_pipe = make_pipeline(SimpleImputer(strategy='median') , StandardScaler())
pipe = num_pipe.fit_transform(housing_num)

cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent') , OneHotEncoder(handle_unknown='ignore'))
'''
this two way to create data frame 
print(pd.DataFrame(pipe  , columns=housing_num.columns , index= housing_num.index))
print('#' * 80)
print(pd.DataFrame(pipe,columns=housing_pipe.get_feature_names_out() , index= housing_num.index))
'''
def try_by_myself_to_convect_cat_to_num(): # by my self
    housing_pipe = make_pipeline(SimpleImputer(strategy='median') , StandardScaler())   
    pipe = housing_pipe.fit_transform(housing_num)
    in_final_housing = pd.DataFrame(pipe , columns= housing_pipe.get_feature_names_out()) #
    text_num_pipe = make_pipeline(OneHotEncoder()) #
    new_housing_with_ocean = pd.DataFrame(text_num_pipe.fit_transform(housing_cat).toarray() , columns=text_num_pipe.get_feature_names_out()) #
    final_housing = pd.concat([in_final_housing , new_housing_with_ocean] , axis=1) #
    return final_housing

def mixed_tranformer_data_using_column_transformer(): # harder
    preprocessing = ColumnTransformer([('num' , num_pipe , num_attribs) , ('cat' , cat_pipe , cat_attribs)])
    inial_data= preprocessing.fit_transform(housing)
    names = [_[5:] for _ in preprocessing.get_feature_names_out()]
    data = pd.DataFrame(inial_data , columns=names)
    print(data)

def mixed_tranformer_data_using_another(): # easier 
    preprocessing = make_column_transformer(
        (num_pipe , make_column_selector(dtype_include=np.number)),
        (cat_pipe ,make_column_selector(dtype_include=object))
        )
    initial_data = preprocessing.fit_transform(housing)
    data = pd.DataFrame(initial_data , columns=preprocessing.get_feature_names_out())
    print(data)
    print(preprocessing.get_feature_names_out())
#__________________________________________________________________--final_procssing_on_data ( final level you , if you understand some things you can get start from here)
def column_raito(X): 
    return X[:,[0]] / X[:,[1]] # divide column  by column
def raito(func_transformer , feature_names_in):
    return ['raito'] # rename 
def raito_pipline(): # this func will replace any missing value then divide columns then then scaled values
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        FunctionTransformer(column_raito , feature_names_out=raito),
        StandardScaler())
log_pipe = make_pipeline( # this pipeline will take log for any column will be join
    SimpleImputer(strategy='mean'),
    FunctionTransformer(np.log , feature_names_out='one-to-one'),
    StandardScaler())
cluster_simil = ClusterSimilary(random_state=42)

default_process = make_pipeline( # this is basic pipeline any columns have no any  processing
    SimpleImputer(strategy='mean'),
    StandardScaler())


preprocessing = ColumnTransformer( # this the main pipeline coulmn it will do all process on all columns
    [('rooms_per_house' , raito_pipline() ,['total_rooms' , 'households']),
     ('bedrooms_raito' , raito_pipline() ,['total_bedrooms' ,'total_rooms']),
     ('people_per_house' , raito_pipline() , ['population' ,'households' ]),
     ('log' , log_pipe , ['total_rooms' , 'total_bedrooms' , 'population' , 'households' , 'median_income']),
     ('geo' , cluster_simil , ['longitude' , 'latitude']),
     ('cat' , cat_pipe ,make_column_selector(dtype_include=object))],
       remainder=default_process)



'''
# try linear_regression
lin_reg = make_pipeline(preprocessing , LinearRegression())
lin_reg.fit(housing , housing_labels)
housing_prediction = lin_reg.predict(housing)
lin_mean_error = mean_squared_error(housing_labels , housing_prediction , squared=False)
'''#-----
'''
#try decision tree
tree_reg = make_pipeline(preprocessing , DecisionTreeRegressor())
tree_reg.fit(housing, housing_labels)
tree_prediction = tree_reg.predict(housing) 
tree_mean_error = mean_squared_error(housing_labels , tree_prediction , squared=False)
tree_rmese = -cross_val_score(tree_reg , housing , housing_labels , scoring='neg_root_mean_squared_error' , cv=10 )
pd.Series(tree_rmese).describe()
'''
(
#print(housing_prediction[:5].round(-2))

#print(tree_prediction[:5].round(-2))

#print(housing_labels.iloc[:5].values)
#print(lin_mean.round(-2))
)
'''
forest_reg = make_pipeline(preprocessing , RandomForestRegressor(random_state=42))
forest_rmese = -cross_val_score(forest_reg , housing , housing_labels , scoring='neg_root_mean_squared_error' , cv=3 )
'''
#-----------------------------------------------adjust par_______________
from sklearn.feature_selection import SelectFromModel

final_pip_line = Pipeline([
    ('preprocessing' , preprocessing) ,('random_forest',RandomForestRegressor(random_state=42))])
param_grid = [
    {'preprocessing__geo__n_clusters':[10,140], 'random_forest__max_features':[12 ,18, 25]}, 
]
param_random = [
    {'preprocessing__geo__n_clusters':randint(low=3 , high = 124), 'random_forest__max_features':randint(low=2 , high = 20)}, 
]
# grid seacrh : this will take some hyperparm and will try all with each others and will give you best model as we see in our case this grid will do 2*3=6
def grid_search_():
    grid_seacrh = GridSearchCV(final_pip_line , param_grid , cv=3 , scoring='neg_root_mean_squared_error' )
    grid_seacrh.fit(housing , housing_labels)
    print(grid_seacrh.best_params_)
    print(grid_seacrh.best_estimator_) 
    print(-grid_seacrh.best_score_.round(-2)) # 41700 
def randomize_serach(): # the same concept like grid but this try random
    random_serach = RandomizedSearchCV(final_pip_line , param_random , n_iter=10 , cv=3 , random_state=42 , scoring='neg_root_mean_squared_error')
    random_serach.fit(housing , housing_labels)
    print(random_serach.best_params_)
    print(random_serach.best_score_)
    return random_serach.best_estimator_

final_model = randomize_serach()
'''
rig_f_m = final_model['random_forest'].feature_importances_

x_test = test_set.drop('median_house_value' , axis = 1)
y_test = test_set['median_house_value'].copy()

first_p_y = final_model.predict(x_test)
'''
# __________________________________________________________confidence interval-
'''
confidence = 0.95
squared_errors = (first_p_y - y_test) **  2
print(f'len of squared_errors:{len(squared_errors)} \n and some data in it :{squared_errors.iloc[:5].values}')
with_sqrt = np.sqrt(t.interval(confidence , len(squared_errors)-1 , loc = squared_errors.mean() , scale = sem(squared_errors)))
with_out_sqrt = t.interval(confidence , len(squared_errors)-1 , loc = squared_errors.mean() , scale = sem(squared_errors))
print(f'with_sqrt :{with_sqrt} \n with_out_sqrt : {with_out_sqrt}')
print(mean_squared_error(y_test , first_p_y))
'''
#--------------------------------------------------------exercies_____
#1 = try support vector machine class (sklearn.svm.SVR) & 2
'''
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
final_pip_line_svr = Pipeline([
    ('preprocessing' , preprocessing),
    ('selector' , SelectFromModel(RandomForestRegressor(random_state=42),threshold = 0.005 )),
    ('svr' , SVR())
])
param_rand_svr = [
    {'svr__kernel':['rbf' , 'linear'] ,'svr__C': randint(low=20 , high = 15000), 'svr__gamma':randint(low=0.01 , high = 1) , 'preprocessing__geo__n_clusters':randint(low=3 , high = 124),}
]
random_search = RandomizedSearchCV(final_pip_line_svr , param_rand_svr , n_iter=10 , cv=3 , random_state=42 , scoring='neg_root_mean_squared_error')
random_search.fit(housing.iloc[:5000] , housing_labels.iloc[:5000])
print(random_search.best_score_)
print(random_search.best_params_)
'''












