import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statistics
from sklearn.metrics import mean_absolute_error
import itertools
from scipy import stats
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler

link = "https://github.com/SimonBrandtS/StackOverflowData/raw/master/survey_results_public.csv"   # Link to our public GitHub, containing the CSV file
df = pd.read_csv(link, error_bad_lines=False)                                                     # Loading it into a Panda Dataframe


# Splitting into a list
def langToList(x):
     t = x.split(";")
     t.sort()
     return t

#Creates a Panda series containing the cut-off values for a given input colu
def createPercenttiles(x):
    xTmp = x["ConvertedComp"]
    QuantileRange = np.arange(0.0, 1.01, 0.01).tolist()
    dfPercentiles = xTmp.quantile(q=QuantileRange)
    return dfPercentiles

#Create a list of label values for each quantile. Represented as a numeric range for finansial compensation for each quantile
def createLabelrange(Qr):
    tmpList = Qr.tolist()
    LabelRange = []
    i = 0
    for x in tmpList:
        if i <100:
            tmpValue = str(int(tmpList[i]))+"-"+str(int(tmpList[i+1])-1)+" USD pr. Year"
            LabelRange.append(tmpValue)
            i += 1
    return LabelRange

#Look up the numeric range for a quantile.
def lookUpLabelRange(x):
    try:
        intX = int(x)
        rangeString = labelNumericRangesPrQuantiles[intX-1]
    except:
        pass
    return rangeString



# Naive Bayes specific pre-processing
df_NB = df[df.ConvertedComp != 0.0].dropna(subset=['ConvertedComp'])          # Creates a dataframe only containing the rows of which the values in ["ConvertedComp"] is larger than 1 and not null
df_NB.reset_index(drop=True, inplace=True)                                    # Resets the index values
df_NB = df_NB[['Respondent','LanguageWorkedWith', 'ConvertedComp','Country']] # Dropping all irrelevant columns
df_NB = df_NB.dropna()                                                        # Dropping all rows containing null values
df_NB.reset_index(drop=True, inplace=True)                                    # Resets the index values
df_NB["LangList"] = df_NB["LanguageWorkedWith"].apply(langToList)             # Split the string object value for Languages into a python list of the same values
df_NB = df_NB.drop(["LanguageWorkedWith"], axis = 1)                          # Dropping the ["LanguageWorkedWith"] column from the data-set


#Parameter tuning #
QuantilesRanges = createPercenttiles(df_NB)                                   # Creates a Panda Series with cut-off values for each Quantile
labelsForQuantiles = np.arange(1.0, 101.0, 1.00).tolist()                     # Creates the correspoding quantile label names for each cut-off values
labelNumericRangesPrQuantiles = createLabelrange(QuantilesRanges)             # Creates the corresponding numerical range for each quantiles
df_NB["QuantilesConvertedComp - in %"] =pd.cut(df_NB["ConvertedComp"], bins=QuantilesRanges, labels=labelsForQuantiles, include_lowest = True) # Creates a new column based on ConvertedComp
                                                                                                                                               # setting the bins to our cut-off values
                                                                                                                                               # and setting our labes to their labels.
df_NB["QuantileNumericRange"] = df_NB["QuantilesConvertedComp - in %"].apply(lookUpLabelRange)                                                 # Creates a new column with the numeric
                                                                                                                                               # range for each quantile value


#Splitting Data into test and training
train, test = train_test_split(df_NB, test_size=0.2)                          # Splitting our dataframe into a train and test set

### Training function for Naïve Bayes ###
def trainModel(data):
    model = {}
    i = 0
    # Mapping for each mixture of langauges in dataframe
    for x in data["LangList"]:
        cont = data["Country"].iloc[i]                                        # Correpsponding country
        wage = data["QuantilesConvertedComp - in %"].iloc[i]                  # Corresponding wage in wuantiles
        for lang in x:                                                        # For each language in the list
            if lang not in model:                                             # If the Language is not in the dictionary, add it and map it to a new dictionary
                model[lang] = {}
                model[lang][cont] = {}                                        # Add contry to the dictionary as a key, and map it to a third dictionary
                model[lang][cont]['observed'] = 1                             # Add an observed key, and map it to one (keeping track of how many times this language/contry combination has been added)
                model[lang][cont][wage] = 1                                   # Map key: wage, to how many times this lang/cont/wage combination has been observed
                
            else:
                if cont not in model[lang]:                                   # if Langauge/country combination does not exist
                    model[lang][cont] = {}
                    
                    if ('observed' not in model[lang][cont]):                 # if langauge/country combination has not been observed
                        model[lang][cont]['observed'] = 1                     # Add 'observed' to combination
                    else:
                        model[lang][cont]['observed'] += 1                    # else, increment observed by one

                    model[lang][cont][wage] = 1                               #Set value of combination to 1
                    
                else:
                    if wage not in model[lang][cont]:                         # if wage not in lang/cont combination
                        model[lang][cont][wage] = 1                           # Set combination to one
                        if('observed' not in cont in model[lang][cont]):    
                            model[lang][cont]['observed'] = 1                 # If lang/cont has not been observed before, set observed to one
                        else:
                            model[lang][cont]['observed'] += 1                # Else, increment with one
                    else:
                        model[lang][cont]['observed'] += 1                    # if the language/country/wage combination has been observed before increment 'observed' and 'wage' with one
                        model[lang][cont][wage] += 1
                       
        i +=1
    return model # retrun trained data

#Training using Naïve Bayes
trained = trainModel(train)
print(trained)

def predict(lang,country, model):
    likelyhoods = {}
    # Calculating likelyhoods based on each observed wage divided by total observed wages for these specific langauges.
    for x in model[lang][country]:
        if(x != 'observed'):
            likelyhoods[x] = model[lang][country][x]/model[lang][country]['observed']
    # Empty list to return best estimate
    estimate = ['', 0]
    for x in likelyhoods:

        # update the most likely wage
        if likelyhoods[x] > estimate[1]:
            estimate = [x, likelyhoods[x]]

        # if the wage is identical to the current most likely wage:
        elif(likelyhoods[x] == estimate[1]):
            estimate.append(x)
            estimate.append(likelyhoods[x]) 

    return estimate


#Calculating the average
def average(lst):
   x = 0
   #remove all instances of strings
   for i in lst:
      if(isinstance(i,str)):
        lst.remove(i)
      else:
        x = x+i

    #if the list is not empty
   if(len(lst)!=0):
        y = len(lst)
        return x/y

# Runs prediction on all data in the xTest set, and returns each respondent's most class
def fullAnalysis(xTrain, xTest):
    i = 0
    results = {}
    analyzedDF = pd.DataFrame({"Respondent":[],"Predicted":[],"Actual":[],"Difference":[]})
    for x in xTest["Respondent"]:                                   # For each respondent
        wages = []
        for z in xTest["LangList"].iloc[i]:                         #For each language per respondent
            try:                                                    #try/except to avoid testing on data that's not in training set
                wage = predict(z,xTest["Country"].iloc[i],xTrain)   # Run predcition
            except:
                pass                                                # Pass if fails 
            wages.append(wage[0])                                   #Add wage to wages

        results[x] = {}
        try:                                                        # Try/catch to avoid errors in median calculation (mostly relevant for calculating average test cases, to avoid 'divide by 0' exceptions)
            pre = statistics.mean(wages)                            #calculate the median wage
            act = xTest["QuantilesConvertedComp - in %"].iloc[i]
            dif = pre-act
            analyzedDF.loc[i] = [x,pre,act,dif]                     # Add prediction, actual comp, and difference between the two to Pandas Dataframe
        except:
            pass
        i+=1
    return analyzedDF


analyzedData = fullAnalysis(trained,test)
analyzedData.reset_index(drop=True, inplace=True)
analyzedData.reset_index(inplace=True)
MAE = mean_absolute_error(analyzedData.Actual,analyzedData.Predicted)
print("MAE:",MAE)

# Visuals #

x = analyzedData.index
y = analyzedData.Difference
plt.scatter(x, y, c="g", alpha=0.045)
plt.xlabel("Respondent_ID")
plt.ylabel("Difference i Quantiles")
plt.title("Using Mean")
plt.show()

NB_Difference = analyzedData["Difference"]
NB_Difference.hist(bins = 100)

plot = NB_Difference.hist(bins = 100)
plot.set_xlabel("Quantiles_Differences")
plot.set_ylabel("Amount")
plt.suptitle("MAE: "+str(MAE))
plt.title("Tuned Classifier")