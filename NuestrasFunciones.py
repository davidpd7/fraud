# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:49:08 2022

@author: Guille
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import winsorize
import scipy.stats as stats
from scipy.stats import f_oneway, chi2_contingency
import patsy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score, silhouette_score
import missingno as msno

from IPython.display import display
from optbinning import OptimalBinning

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from scipy.spatial.distance import cdist, pdist


""" DEPURACION Y VISUALIZACION DE DATOS """

## Funcion para covertir a categoricas aquellas variables con menos de 10 valores unicos


def categorizer(data):
    """
    Imput: DataFrame
    
    """
    to_factor = list(data.loc[:,data.nunique() < 10])
    data[to_factor] = data[to_factor].astype('category') 


## Función para histograma con boxplot 
def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()
    
## Función para gráfico de barras de variables categóricas
def cat_plot(col):
    if col.dtypes == 'category':
        sns.countplot(x=col)
        plt.show()


## Función general plot para aplicar al archivo por columnas
def plot(col):
     if col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_plot(col)

## Función manual de winsor con clip+quantile 
def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

## Función para gestionar outliers
def gestiona_outliers(col,clas = 'check'):
    
     print(col.name)
     # Condición de asimetría y aplicación de criterio 1 según el caso
     if abs(col.skew()) < 1:
        criterio1 = abs((col-col.mean())/col.std())>3
     else:
        criterio1 = abs((col-col.median())/col.mad())>8
     
     # Calcular primer cuartil     
     q1 = col.quantile(0.25)  
     # Calcular tercer cuartil  
     q3 = col.quantile(0.75)
     # Calculo de IQR
     IQR=q3-q1
     # Calcular criterio 2 (general para cualquier asimetría)
     criterio2 = (col<(q1 - 3*IQR))|(col>(q3 + 3*IQR))
     lower = col[criterio1&criterio2&(col<q1)].count()/col.dropna().count()
     upper = col[criterio1&criterio2&(col>q3)].count()/col.dropna().count()
     # Salida según el tipo deseado
     if clas == 'check':
            return(lower*100,upper*100,(lower+upper)*100)
     elif clas == 'winsor':
            return(winsorize_with_pandas(col,(lower,upper)))
     elif clas == 'miss':
            print('\n MissingAntes: ' + str(col.isna().sum()))
            col.loc[criterio1&criterio2] = np.nan
            print('MissingDespues: ' + str(col.isna().sum()) +'\n')
            return(col)
     
def missing_values(data):
    msno.matrix(data, figsize = (30,10))
    df = pd.DataFrame()
    for col in data.columns:
        unique_values = data[col].unique()
        try:
            unique_values = np.sort(unique_values)
        except:
            pass
        nans = round(pd.isnull(data[col]).sum()/data.shape[0]*100, 1)
        zeros = round((data[col] == 0).sum()/data.shape[0]*100, 1)
        #empty = round((data[data[col]] == '').sum()/data.shape[0]*100, 1)
        df = df.append(pd.DataFrame([col,
                                     len(unique_values),
                                     nans,
                                     zeros,
                                     #empty
                                    ]).T, ignore_index = True)

    return df.rename(columns = {0: 'variable',
                                1: 'Unique values',
                                2: 'Nan %',
                                3: 'zeros %',
                                #4: 'empty'
                               }).sort_values('Nan %', ascending=False)



# Función para calcular VCramer (dos nominales de entrada!)
def cramers_v(var1, varObj):
    if not var1.dtypes == 'category':
        #bins = min(5,var1.value_counts().count())
        var1 = pd.cut(var1, bins = 5)
    if not varObj.dtypes == 'category': #np.issubdtype(varObj, np.number):
        #bins = min(5,varObj.value_counts().count())
        varObj = pd.cut(varObj, bins = 5)
        
    data = pd.crosstab(var1, varObj).values
    vCramer = stats.contingency.association(data, method = 'cramer')
    return vCramer


# Ejemplo uso univariante
#cramers_v(vinosCompra['Etiqueta'],vinosCompra['Beneficio'])

# Aplicar la función al input completo contra la objetivo
#tablaCramer = pd.DataFrame(imputCompra.apply(lambda x: cramers_v(x,varObjCont)),columns=['VCramer'])


## Función mejor tranformación ##
# Busca la transformación de variables input de intervalo que maximiza la VCramer o 
# la correlación tipo Pearson con la objetivo
def mejorTransf (vv,target, name=False, tipo = 'cramer', graf=False):
    
    # Escalado de datos (evitar fallos de tamaño de float64 al hacer exp de número grande..cosas de python)
    vv = pd.Series(scale(vv), name=vv.name)
    # Traslación a valores positivos de la variable (sino falla log y las raíces!)
    vv = vv + abs(min(vv))+0.0001
      
    # Definimos y calculamos las tranformacione típicas  
    transf = pd.DataFrame({vv.name + '_ident': vv, vv.name + '_log': np.log(vv), vv.name + '_exp': np.exp(vv), 
                         vv.name + '_sqr': np.square(vv), vv.name + '_cuarta': vv**4, vv.name + '_raiz4': vv**(1/4)})
      
    # Distinguimos caso cramer o caso correlación
    if tipo == 'cramer':
      # Aplicar la función cramers_v a cada trasnformación frente a la respuesta
      tablaCramer = pd.DataFrame(transf.apply(lambda x: cramers_v(x,target)),columns=['VCramer'])
      
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCramer,x=tablaCramer.VCramer,title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # Identificar mejor transfromación
      best = tablaCramer.query('VCramer == VCramer.max()').index
      ser = transf[best[0]].squeeze()
    
    if tipo == 'cor':
      # Aplicar coeficiente de correlacion a cada trasnformación frente a la respuesta
      tablaCorr = pd.DataFrame(transf.apply(lambda x: np.corrcoef(x,target)[0,1]),columns=['Corr'])
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCorr,x=tablaCorr.Corr,title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # identificar mejor transfromación
      best = tablaCorr.query('Corr.abs() == Corr.abs().max()').index
      ser = transf[best[0]].squeeze()
  
    # Aquí distingue si se devuelve la variable transfromada o solamente el nombre de la transfromacion
    if name:
      return(ser.name)
    else:
      return(ser)

# Ejemplo de uso univariante
#tr = mejorTransf(vinosCompra.Azucar,varObjCont, tipo='cor')


"""    REGRESIONES Y MODELIZACIÓN     """


# Función para generar la fórmula por larga que sea
def ols_formula(df, dependent_var, *excluded_cols):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)

# Aplicamos a fórmula de modelo completo
#form=ols_formula(data_train,'Beneficio')

# Función para evaluación de modelos Logísticos en training-test (input: fórmula y dataset al natural)
def tr_tst_eval_lin(formula,data):
  # Generamos las matrices de diseño según la fórmula de modelo completo
  y, X = patsy.dmatrices(formula, data, return_type='dataframe')
  
  # Creamos 4 objetos: predictores para tr y tst y variable objetivo para tr y tst. 
  X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=1234)
  
  # Definición de modelo
  modelo =  LinearRegression()
  
  # Ajuste de modelo
  modelo = modelo.fit(X_tr,y_tr)
  
  # Accuracy del modelo en training
  r2 = modelo.score(X_tr,y_tr)
  print('Coeficiente de determinación TRAINING: ',r2, '\n')
  
  # Predicciones en test
  y_pred = modelo.predict(X_tst)

  # Cálculo de performance
  print("Mean squared error TEST: %.2f" % np.sqrt(mean_squared_error(y_tst, y_pred)))
  print("Coeficiente de determinación TEST: %.2f" % r2_score(y_tst, y_pred))


# Función para evaluación de modelos Logísticos en training-test (input: fórmula y dataset al natural)
def tr_tst_eval_log(formula,data):
  # Generamos las matrices de diseño según la fórmula de modelo completo
  y, X = patsy.dmatrices(formula, data, return_type='dataframe')
  
  # Creamos 4 objetos: predictores para tr y tst y variable objetivo para tr y tst. 
  X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=1234)
  
  # Definición de modelo
  modelo = LogisticRegression(solver='lbfgs', max_iter=2000,penalty='none')
  
  # Arreglar y para que le guste a sklearn...numeric
  y_tr_ = y_tr.iloc[:,0].ravel()
  
  # Ajuste de modelo
  modelLog = modelo.fit(X_tr,y_tr_)
  
  # Accuracy del modelo en training
  acc = modelLog.score(X_tr,y_tr_)
  print('Accuracy en training: ',acc, '\n')
  
  # Predicciones en test
  y_pred = modelLog.predict(X_tst)
  
  # Matriz de confusion de clasificación 
  print('Matriz de confusión y métricas derivadas: \n',confusion_matrix(y_tst,y_pred))
  
  # Reporte de clasificación 
  print(metrics.classification_report(y_tst,y_pred))
  
  # Extraemos el Area bajo la curva ROC
  print('Area bajo la curva ROC training: \n', metrics.roc_auc_score(y_tr, modelLog.predict_proba(X_tr)[:, 1]))
  print('Area bajo la curva ROC test: \n', metrics.roc_auc_score(y_tst, modelLog.predict_proba(X_tst)[:, 1]))


# Función para comparación por validación cruzada
def cross_val_lin(formula, data, seed=12345):
      # Generamos las matrices de diseño según la fórmula de modelo completo
      y, X = patsy.dmatrices(formula, data, return_type='dataframe')
      
      model = LinearRegression()
      
      # Establecemos esquema de validación fijando random_state (reproducibilidad)
      cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)
  
      # Obtenemos los resultados de R2 para cada partición tr-tst
      scores = cross_val_score(model, X, y, cv=cv)
  
      # Sesgo y varianza
      print('Modelo: ' + formula)
      print('Coeficiente de determinación R2: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
      
      #sns.violinplot(y=scores,palette='viridis')
      
      return(scores)

# Función para comparación por validación cruzada
def cross_val_log(formula, data, seed=12345):
    # Generamos las matrices de diseño según la fórmula de modelo completo
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    y = y.iloc[:,0].ravel()
      
    model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='none')
  
    # Establecemos esquema de validación fijando random_state (reproducibilidad)
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)
     
    # metrics.get_scorer_names() --> Posibilidades de distintas métricas! 
      
    # Obtenemos los resultados de R2 para cada partición tr-tst
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
  
    # Sesgo y varianza
    print('Modelo: ' + formula)
    print('AUC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
      
    #sns.violinplot(y=scores,palette='viridis')
      
    return(scores)
  

  
# Función para pintar la curva ROC
def roc_grafico(test,pred): 
    fpr, tpr, thresholds = roc_curve(test,pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
 
# Función pto de corte por Youden
def cutoff_youden(test,pred):
    fpr, tpr, thresholds = roc_curve(test,pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

# Función para comparación por validación cruzada válido para selección
# de variables por sfs o Lasso o clualquier tipo de dataset reducido en número de variables.
def cross_val_selectVar(sfs, data, y, log=False, seed=12345):
        # Por defecto, digamos que la entrada es una matriz explícita de variables seleccionadas
        X = sfs
	
	# Si realmente es un objeto sfs, obtenemos matriz explícita con transform
        if not isinstance(sfs,pd.DataFrame):
            X = sfs.transform(data)
        # Para logística arreglar variable objetivo para sklearn, definit logistic y métrica    
        if log: 
            y.iloc[:,0].ravel()
            model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='none')
            scoring='roc_auc'
        # En caso contrario es regresión lineal y lo especificamos
        else: 
            model = LinearRegression()
            scoring='r2'
        
        # Establecemos esquema de validación fijando random_state (reproducibilidad)
        cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)

        # Obtenemos los resultados de R2 para cada partición tr-tst
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)

        # Sesgo y varianza
        print('Métrica ' + scoring + ': %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

       # sns.violinplot(y=scores,palette='viridis')

        return(scores)
    

"""       SERIES TEMPORALES     """


# Función para pasar el test de estacionariedad de Dickey Fuller
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# Función para pintar el gráfico estacional
def seasonal_plot(df, season='year', index='month', column=None):
    """Makes a seasonal plot of one column of the input dataframe. Considers the first columns by default.
    
    Arguments:
    
    - df (Pandas DataFrame): DataFrame indexed by Datetime (see `parse_dates` parameter when reading a CSV);
    - season (string): the season that you want to considering when doing the plot, e.g., year, month, etc.;
    - index (string): corresponds to the X axis of the plot. You should choose based on the index period that you're using;
    - column (string, optional): the DataFrame column to consider. Picks the first one by default.
    """
    if column == None:
        column = df.columns[0]
        
    piv_index = getattr(df.index, index)
    piv_season = getattr(df.index, season)
    
    piv = pd.pivot_table(df, index=piv_index, columns=piv_season, values=[column])
    piv.plot(figsize=(12,8))
    

# Función para evaluar modelo de suavizado o arima manual (No válido para auto_arima)
def eval_model(model,tr,tst,name='Model',lags=12):
    lb = np.mean(sm.stats.acorr_ljungbox(model.resid, lags=lags, return_df=True).lb_pvalue)
    pred = model.forecast(steps=len(tst))
    fig1, ax = plt.subplots()
    ax.plot(tr, label='training')
    ax.plot(tst, label='test')
    ax.plot(pred, label='prediction')
    plt.legend(loc='upper left')
    tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(round(mean_absolute_percentage_error(tst, pred)*100,2)) + "%"
    plt.title(tit)
    plt.ylabel('Pasajeros')
    plt.xlabel('Date')
    plt.show()
    print(lb)
 
# Función para evaluar modelo auto_arima
def eval_model_Aarima(model,tr,tst,name='Model',lags=12):
      lb = np.mean(sm.stats.acorr_ljungbox(model.resid(), lags=lags, return_df=True).lb_pvalue)
      pred = model.predict(n_periods=len(tst))
      fig1, ax = plt.subplots()
      ax.plot(tr, label='training')
      ax.plot(tst, label='test')
      ax.plot(pred, label='prediction')
      plt.legend(loc='upper left')
      tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(round(mean_absolute_percentage_error(tst, pred)*100,2)) + "%"
      plt.title(tit)
      plt.ylabel('Serie')
      plt.xlabel('Date')
      plt.show()
      model.plot_diagnostics(figsize=(14,10))
      plt.show()  


# Función para evaluar residuos a través de contrastes de hipótesis
def residcheck(residuals, lags):
    """
    Function to check if the residuals are white noise. Ideally the residuals should be uncorrelated, zero mean, 
    constant variance and normally distributed. First two are must, while last two are good to have. 
    If the first two are not met, we have not fully captured the information from the data for prediction. 
    Consider different model and/or add exogenous variable. 
        
    If Ljung Box test shows p> 0.05, the residuals as a group are white noise. Some lags might still be significant. 
        
    Lags should be min(2*seasonal_period, T/5)
        
    plots from: https://tomaugspurger.github.io/modern-7-timeseries.html
        
    """
    resid_mean = np.mean(residuals)
    lj_p_val = np.mean(sm.stats.acorr_ljungbox(x=residuals, lags=lags).lb_pvalue)
    norm_p_val =  stats.jarque_bera(residuals)[1]
    adfuller_p = adfuller(residuals)[1]
        
      
    fig = plt.figure(figsize=(10,8))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2);
    acf_ax = plt.subplot2grid(layout, (1, 0));
    kde_ax = plt.subplot2grid(layout, (1, 1));
    
    residuals.plot(ax=ts_ax)
    plot_acf(residuals, lags=lags, ax=acf_ax);
    sns.kdeplot(residuals);
    #[ax.set_xlim(1.5) for ax in [acf_ax, kde_ax]]
    sns.despine()
    plt.tight_layout();
    plt.show()
    print("** Mean of the residuals: ", np.around(resid_mean,2))
        
    print("\n** Ljung Box Test, p-value:", np.around(lj_p_val,3), 
        "(>0.05, Uncorrelated)" if (lj_p_val > 0.05) else "(<0.05, Correlated)")
        
    print("\n** Jarque Bera Normality Test, p_value:", np.around(norm_p_val,3),
        "(>0.05, Normal)" if (norm_p_val>0.05) else "(<0.05, Not-normal)")
        
    print("\n** AD Fuller, p_value:", np.around(adfuller_p,3), 
        "(>0.05, Non-stationary)" if (adfuller_p > 0.05) else "(<0.05, Stationary)")
    
    return ts_ax, acf_ax, kde_ax   


"""    REDUCCION DE DIMENSIONES  """

# Dibujar biplot 
def biplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley) #, c = cities.index.tolist())
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

# Ejemplo de uso. Siempre para 2 componentes!
#biplot(scores[:,0:2],np.transpose(pca.components_[0:2, :]),cities.columns)
#plt.show()


"""    CLUSTERING    """

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from time import time

# Extraer dendograma y pintarlo
def plot_dendogram(model, **kwargs):
  '''
  Esta función extrae la información de un modelo AgglomerativeClustering
  y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
  '''
      
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1  # leaf node
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count
  
  linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)
  
  # Plot
  dendrogram(linkage_matrix, **kwargs)
  plt.show()
  
  
# Cremos función scree_plot_kmeans para buscar el número de clusters óptimo 
# con 3 métricas usuales. Wss, silueta y % de varianza explicada
# Input: data = dataset en bruto (se escala dentro de la propia función)
#        n_max = número máximo de grupos a evaluar
# ==============================================================================

def scree_plot_kmeans(data,n_max):
  range_n_clusters = range(2, n_max)
  X_scaled = scale(data)
  inertias = []
  silhouette = []
  var_perc = []
  
  for n_clusters in range_n_clusters:
      modelo_kmeans = KMeans(
                          n_clusters   = n_clusters, 
                          n_init       = 20, 
                          random_state = 123
                      )
      modelo_kmeans.fit(X_scaled)
      cluster_labels = modelo_kmeans.fit_predict(X_scaled)
      inertias.append(modelo_kmeans.inertia_)
      silhouette.append(silhouette_score(X_scaled, cluster_labels))
      tss = sum(pdist(X_scaled)**2)/X_scaled.shape[0]
      bss = tss - modelo_kmeans.inertia_
      var_perc.append(bss/tss*100)
      
  fig, ax = plt.subplots(1, 3, figsize=(16, 6))
  ax[0].plot(range_n_clusters, inertias, marker='o')
  ax[0].set_title("Scree plot Varianza intra")
  ax[0].set_xlabel('Número clusters')
  ax[0].set_ylabel('Intra-cluster (inertia)')
  
  ax[1].plot(range_n_clusters, silhouette, marker='o')
  ax[1].set_title("Scree plot silhouette")
  ax[1].set_xlabel('Número clusters')
  ax[1].set_ylabel('Media índices silhouette');
  
  ax[2].plot(range_n_clusters, var_perc, marker='o')
  ax[2].set_title("Scree plot % Varianza")
  ax[2].set_xlabel('Número clusters')
  ax[2].set_ylabel('% de varianza explicada')


def cat_relational_study(data, VarObj, column):
    potential_var_cat = []
    
    print(f'Value Weight Column {column} - Normalize = False')
    crosstab_n1 = pd.crosstab(data[column], data[VarObj], margins=True, normalize=1).round(3)
    display(crosstab_n1)
    
    print(f'Value Weight Column {column} - Normalize = True')
    crosstab_n0 = pd.crosstab(data[column], data[VarObj], margins=True, normalize=0).round(3)
    display(crosstab_n0)
    print(f'Cross Tab {column}')
    ctabla = pd.crosstab(data[column], data[VarObj], margins=True).round(3)
    display(ctabla)
    
    c, p, dof, expected = chi2_contingency(ctabla) 
    print(f'P value {column}: {p}')
    if p < 0.05:
        print(f'{column} Significant, candidate to include in Modele')
        potential_var_cat.append(column)
    else:
        print('P-Value is not significant.')
   
    
    print('Cross Tab Graph - Normalize = 0')
    crosstab_n0.plot(kind='bar', figsize=(15, 5))
    plt.show()
    
    print('Cross Tab Graph - Normalize = 1')
    crosstab_n1.plot(kind = 'bar',figsize=(15, 5))
    plt.show()
    return print(potential_var_cat)
  
def cont_relational_study(data, column, VarObj):
    print('Variable normal')
    sns.displot(x=(data[column]), kind="kde", fill=True)
    plt.show()
    print('Variable log')
    sns.displot(x=np.log(data[column]), kind="kde", fill=True)
    plt.show()
    sns.displot(x=(data[column]), kind="kde", fill=True, hue=data[VarObj], common_norm=False)
    plt.show()
    fvalue, pvalue = f_oneway(data.loc[data[VarObj]==0,[column]], data.loc[data[VarObj]==1,[column]])
    print(f'Contraste de diferencia de medias:{fvalue, pvalue}')
    if pvalue < 0.05:
        print('Rechazo hipotesis nula, existen diferencia de medias')
    else:
        print('Aceptacion hipotesis nula, no existen diferencia de medias')

def scoring_analasis(data, columns, VarObj):
    for col in columns: 
        if data[col].dtypes != 'category':
            cont_relational_study(data, col, VarObj)
        else:
            cat_relational_study(data, col, VarObj)


def optimal_binning(data_train, variables, varObj, binning_type, dtype):
    for variable in variables:
        Y = data_train[varObj].values
        optb = OptimalBinning(name=variable, dtype=binning_type, solver="cp")
        X = data_train[variable].values
        optb.fit(X, Y)
        optb.splits
        binning_table = optb.binning_table
        a = binning_table.build()
        if binning_table.iv > 0.02:
            print(variable)
            display(a)
            print(f'Gini: {binning_table.gini}')
            print(f'IV: {binning_table.iv}')
            print('WOE Plot')
            binning_table.plot(metric='woe')
            print('Event Rate Plot')
            binning_table.plot(metric='event_rate')
        else:
            print(f'{variable} IV under limit of 0.02')
            print(f'Gini: {binning_table.gini}')
            print(f'IV: {binning_table.iv}')


def confussion_matrix(y, predictions):

    conf_matrix = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def custom_roc_curve( y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

def metrics(model, X, y):
    print(model.__class__.__name__)
    print("------------------------")
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    confussion_matrix(y, predictions)
    print("------------------------")
    custom_roc_curve(y, probs)

def models_train(model, X_train, y_train):

    print(f"{model.__class__.__name__}")
    start_time = time()
    model.fit(X_train, y_train)
    end_time = time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    print("------------------------")

def model_comparison(model_instances, X, y):

    names = []
    accuracy_scores = []
    roc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for model in model_instances:
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        roc_score = roc_auc_score(y, predictions)
        names.append(model.__class__.__name__)
        accuracy_scores.append(accuracy)
        roc_scores.append(roc_score)
        f1_scores.append(f1_score(y, predictions))
        precision_scores.append(precision_score(y, predictions))
        recall_scores.append(recall_score(y, predictions))

    results_df = pd.DataFrame({
          "Model": names,
          "Accuracy Val": accuracy_scores,
          "ROC Score Val": roc_scores,
          "F1 Score Val": f1_scores,
          "Precision Score Val": precision_scores,
          "Recall Score Val": recall_scores,
      })
    return results_df

