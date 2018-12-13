
# coding: utf-8

# In[1]:


spark


# In[2]:


df = spark.read.format("csv").option("header","true").option("inferSchema","true").load("/data/Combined_Cycle_Power_Plant.csv")


# In[3]:


df.show()


# In[4]:


df.cache()


# In[6]:


df.limit(10).toPandas().head()


# In[7]:


from pyspark.ml.feature import *


# In[8]:


vectorizer = VectorAssembler()
vectorizer.setInputCols(["AT", "V", "AP", "RH"])
vectorizer.setOutputCol("features")

df_vect = vectorizer.transform(df)
df_vect.show(10, False)


# In[9]:


print(vectorizer.explainParams())


# In[10]:


from pyspark.ml.regression import LinearRegression


# In[11]:


lr = LinearRegression()
print(lr.explainParams())


# In[12]:




lr.setLabelCol("EP")
lr.setFeaturesCol("features")
model = lr.fit(df_vect)


# In[13]:


type(model)


# In[14]:


print("R2:", model.summary.r2)
print("Intercept: ", model.intercept, "Coefficients", model.coefficients)


# In[15]:


df_pred = model.transform(df_vect)
df_pred.show()


# In[16]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[17]:


evaluator = RegressionEvaluator()
print(evaluator.explainParams())


# In[18]:


evaluator = RegressionEvaluator(labelCol = "EP", 
                                predictionCol = "prediction", 
                                metricName = "rmse")
evaluator.evaluate(df_pred)


# In[19]:


from pyspark.ml.pipeline import Pipeline, PipelineModel


# In[20]:


pipeline = Pipeline()
print(pipeline.explainParams())
pipeline.setStages([vectorizer, lr])
pipelineModel = pipeline.fit(df)


# In[21]:


pipeline.getStages()


# In[22]:


lr_model = pipelineModel.stages[1]
lr_model .coefficients


# In[23]:


pipelineModel.transform(df).show()


# In[24]:


evaluator.evaluate(pipelineModel.transform(df))


# In[ ]:


#export SPARK_HOME=/usr/lib/spark-2.3.0-bin-hadoop2.7
#training@training-VirtualBox:/usr/lib$ export PYSPARK_PYTHON=python3
#training@training-VirtualBox:/usr/lib$ export PYSPARK_DRIVER_PYTHON=jupyter
#training@training-VirtualBox:/usr/lib$ export PYSPARK_DRIVER_PYTHON_OPTS="notebook --NotebookApp.ip='*' --NotebookApp.port=8888 --NotebookApp.open_browser=False"
#training@training-VirtualBox:/usr/lib$ $SPARK_HOME/bin/pyspark

