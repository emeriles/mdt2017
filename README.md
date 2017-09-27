# mdt2017

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
Algunos resultados pueden ser vistos en results.txt