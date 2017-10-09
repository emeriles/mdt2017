# mdt2017


# Parte 1 (Clustering)

[Link a la entrega de la primera parte del proyecto](https://github.com/emeriles/mdt2017/tree/c2d515f9fc74e055762a5d2ea2bd9005ac974730)

[Link a la entrega de la segunda parte del proyecto](https://github.com/emeriles/mdt2017#parte-2-feature-selection)

# Elección del corpus

Debido a las limitaciones materiales de infraestructura y lo pesado del procesamiento, se optó por un corpus de texto plano de aproximadamente 34 MB. Se trata de una recolección de noticias de 35 Mb aproximadamente. Es para éste corpus que se particularizó el código, y se trabajó con fragmentos de hasta 5000 noticias.

# Preprocesamiento del corpus

Se consideró en un principio tokenizar el texto con el PlainTextCorpusReader, siguiendo [estas](http://nbviewer.jupyter.org/url/cs.famaf.unc.edu.ar/~francolq/Tokenizaci%C3%B3n%20con%20NLTK.ipynb) instrucciones. A pesar de lo fácil de la técnica, no se cuenta con una versatilidad necesaria para sacar, por ejemplo, las comunes ocurrencias en el texto de caracteres como "\&#226;\&#8364;\&#8220", que debieron recibir especial atención. Es así que se optó por [LazyLoader](https://kite.com/docs/python/python;nltk.data.LazyLoader), también de nltk que permite separar en pasos más granulares el procesamiento del texto.
Se usaron expresiones regulares para la captura de tokens y para la eliminación de caracteres no deseados y frecuentes en el texto, como el ya mencionado arriba.

# Herramienta de clustering

La herramienta de clustering que se usó es K-Means debido al bajo costo de recursos. La implementación elegida es la de [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Se trata de una librería con muchos parámetros de interés: número de clusters, métodos propios de scikit de optimización para encontrar centroides iniciales, cantidad de veces que el algoritmo se reinicia para encontar mejores opciones de centroides, semilla para controlar la aleatoriedad, entre otros, incluyendo parámetros de paralelización. Para apreciar la relevancia de estos parámetros recordemos que el método de clustering K-Means depende fuertemente de los valores iniciales de los centroides.
Por los resultados obtenidos (es común la aparición de clusters con una palabra), se preferiría aplicar una técnica de clustering por bisecciones, de forma tal que se fuercen relaciones entre palabras.

# Features

Se usó como librería principal para vectorizar palabras la librería [DictVectorizer de scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html).
Los features considerados fueron:
   - POS tagging
   - POS tagging del contexto
   - palabras de contexto que no sean stop words

Los POS tags se obtuvieron del [Stanford POS tagger](https://nlp.stanford.edu/software/tagger.shtml). Se los procesó tomando tramos iniciales de cada tag, para ir obteniendo unidades de información progresivamente más profundas de las palabras.
Los pos tags de contexto fueron considerados sólo en su raiz (palabra inicial).
Las palabras de contexto fueron tratadas con stemming, para prevenir la explosión de dimensionalidades y para poder hacer mejores relaciones entre palabras. En este sentido se probó la forma en la que escalaban los features mencionados y se corroboró que se reducía la dimensionalidad a la mitad:
   - Con 1000 noticias: (y piso de frequencias de palabras=50) 7413 con stemming y 14522 sin stemming.
   - Con 300 noticias: (y piso de frequencias de palabra=50) 5549 con stemming y 9876 sin stemming.

Es interesante observar algunas agrupaciones hechas en éste respecto. Notar las agrupaciones, probablemente más pertinentes. Resultados obtenidos de aprendizaje de 1000 noticias con 80 clusters:
Con stemming:
 - seis cuatro cinco
 - ecuador interior brasil chile
 - casa escuela capital policía
 - tener hacer ver

Sin stemmer:
 - cinco seis cuatro
 - federal chile lula belgrano aires ecuador kirchner correa dilma
 - tribunal escuela capital casa
 - tener ver hacer

La selección de palabras de contexto es aleatoria sobre la oración a la que pertenece la palabra a caracterizar. Posibles mejoras podrían tener en cuenta agregar features pre-procesados por Named Entity Recognition para hacer relaciones todavía más significativas.
En todos los casos trabajados este último parámetro (cantidad de palabras de contexto a sortear) se mantuvo en tres.

# Reducción de dimensionalidad

Además del stemming sobre palabras de contexto, se usó para reducir la dimensionalidad un umbral de frecuencia. Se lo trató a éste como un parámetro más a considerar en todas las corridas.

# Discusión

Se comprendió que en esta técnica de aprendizaje no supervisado, las elecciones de diseño y configuración son muchísimas. Desde la elección de tokenizadores hasta la elección de centroides iniciales para las iteraciones, el problema es amplio. Sin embargo, con las sucesivas corridas del algoritmo se fueron encontrando relaciones importantes dentro de los clústers.
Muchos de los detalles específicos de la solución presentada fueron ya descriptos y ejemplificados.
Algunos resultados pueden ser vistos en `results_1ra_parte.txt`

# Parte 2 (Feature selection)

## Descripción del corpus no anotado

El preproceso y la preparación para el corpus fue exactamente la misma que para la primera parte.

## Descripción de las librerías utilizadas

Para esta segunda parte se usó la librería [Spacy](https://spacy.io/). Es una librería "de fuerza industrial" para procesamiento de lenguaje natural. De ella se pudieron extraer con mucha facilidad una variedad mas amplia de features que en la version pasada (Véase Parte 1)
Otra librería fundamental en el presente trabajo es [scikit-learn](http://scikit-learn.org). De la misma se usaron tanto la técnica principal de K-Means, como así también tecnicas de selección de features y de normalización de matrices.

## Features trabajados

 - POS tag
 - POS tag de palabra previa y siguiente
 - triplas de dependencia con respecto a la raiz de una palabra
 - palabras de contexto en la oración

La forma de implementación de las palabras de contexto fue similar al de la primera parte, mientras que los POS tags esta vez fueron los retornados por Spacy, que tienen un poco menos de riqueza de información en los tags.
Con respecto a las triplas de dependencia, se extrajo, en un mismo feature, la relación de dependencia y la palabra (objetivo) con respecto a la cuál existe tal relación. A su vez, a la palabra objetivo: se la tomó siempre en minúscula y además se la procesó con un _stemmer_ (el mismo que el de la primera parte); y se dejó de lado toda relación de una palabra consigo misma. Todo ello para prevenir explosiones de dimensionalidad.

## Tecnica no supervisada de feature selection

Las técnicas no supervisadas de feature selection consideradas fueron dos:
 -  [_Variance Treshold_](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold)
 -  [_Singular Value Descomposition_](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) (_SVD_)

En ambos casos se hizo antes un proceso de normalización utilizando la [función Normalize de scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html).
Luego de muchos intentos, la ténica de _Variance Treshold_ no arrojó resultados positivos. En efecto, los clusters obtenidos consistían todos "singletones" (cluster que contiene una palabra), exceptuando uno de ellos que contenía al resto de las palabras. Se probó con variar el treshold (desde 0.000000001 hasta 0.01) viendo que claramente las dimensiones de los features se reducían, aunque a pesar de ello, los resultados antes mencionados se matenían.
La técnica que sí arrojó resultados interesantes fue SVD. Se notó una leve mejora con respecto a la no aplicación de SVD.

## Resultados y su discusión

Para apreciar las diferencias veamos clusters donde se repitan palabras similares. Se harán comparaciones entre resultados obtenidos con normalización y reducción de dimensionalidad (Full), y resultados sin ninguna de estas características(Raw).

Clusters de palabras seleccionadas sobre _Raw_:
 - octubre servicio septiembre marzo enero dólares agosto cambio julio
 - san juan
 - sistema decisión plan chicos elección grupo proyecto
 - brasil chile dilma interior ecuador río lula
 - alejandro daniel jerónimo marina rafael instituto cristina hugo néstor santa

Clusters de palabras seleccionadas sobre _Full_:
 - agosto octubre enero julio marzo infraestructura agua septiembre dólares
 - rafael san santa josé hugo daniel jerónimo juan manuel instituto carlos néstor marina cristina villa alejandro
 - conducción acto caso modelo ciudad plan sede encuentro toma estrategia decisión mano resto pedido tipo ley
 - apablaza cabrera río brasil chile ecuador turismo américa lula españa córdoba dilma
 - resultados tomas proyecto zona región vecinos oposición hombre

Si bien se trata de una muestra parcial de los métodos aquí expuestos, se puede ver cómo las técnicas mezclaron palabras entre los clusters. Es así que, tal vez, las mejoras son en realidad parciales. Con respecto al tiempo de ejecución, no se notaron diferencias significativas. Se presume que es a causa de que el alivianamiento de tareas por un lado (el trabajar con menos features a la hora de clusterizar) compensa por otro lado con la tarea de reducir dimensionalidad.