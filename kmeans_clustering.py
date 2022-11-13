import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Importando uma tabela com os dados
data = pd.read_csv('/content/drive/MyDrive/portfolio/ciencia_de_dados/trabalho_2/table_bn.csv')

#Primeira etapa do relatório
datt=data
# Nomeando os eixos 'x' e 'y'
plt.xlabel('V1')
plt.ylabel('V2')
plt.title("Analyzing characteristics", 
          fontdict={'family': 'monospace', 
                    'color' : 'black',
                    'weight': 'bold',
                    'size': 16},
          loc='center')    
#Plotando os pontos num gráfico
plt.scatter(x = data['V1'],y = data['V2'], alpha=0.3,color="yellow")
plt.show()

# Segunda etapa do relatório
# Medindo a média de cada coluna
x_mean=np.mean(datt['V1'])
y_mean=np.mean(datt['V2'])

# Medindo o desvio padrão de cada coluna
x_std=np.std(datt['V1'])
y_std=np.std(datt['V2'])

plt.text(x_mean,y_mean,'Mean', color="red")
print('Mean: (',x_mean,";",y_mean,')')
print("\nStandard deviation: (",x_std,";",y_std,")")

# Nomeando os eixos 'x' e 'y'
plt.xlabel('V1')
plt.ylabel('V2')
plt.title("Analyzing characteristics", 
          fontdict={'family': 'monospace', 
                    'color' : 'black',
                    'weight': 'bold',
                    'size': 16},
          loc='center')    

# Plotando os pontos num gráfico
plt.scatter(x = data['V1'],y = data['V2'], alpha=0.3,color="yellow")
plt.show()

# Terceira etapa do relatório
# Medindo a média de cada coluna
x_mean=np.mean(data['V1'])
y_mean=np.mean(data['V2'])

# Medindo o desvio padrão de cada coluna
x_std=np.std(data['V1'])
y_std=np.std(data['V2'])

# "Padronizando" os dados
data['V1']=((data['V1'])-x_mean)/(x_std)
data['V2']=((data['V2'])-y_mean)/(y_std)

# Centróides
km_res = KMeans(n_clusters = 2).fit(data)
clusters = km_res.cluster_centers_

# Nomeando os eixos 'x' e 'y'
plt.xlabel('V1')
plt.ylabel('V2')
plt.title("Analyzing characteristics", 
          fontdict={'family': 'monospace', 
                    'color' : 'black',
                    'weight': 'bold',
                    'size': 16},
          loc='center')    

# Plotando os pontos num gráfico
plt.scatter(x = data['V1'],y = data['V2'], alpha=0.3,color="yellow")
plt.scatter(clusters[:,0],clusters[:,1],color='red')
plt.show()

#Quarta etapa do relatório

x_mean=np.mean(data['V1'])
y_mean=np.mean(data['V2'])

x_std=np.std(data['V1'])
y_std=np.std(data['V2'])

# Padronização dos dados
data['V1']=((data['V1'])-x_mean)/(x_std)
data['V2']=((data['V2'])-y_mean)/(y_std)

# Centróides
km_res = KMeans(n_clusters = 2).fit(data)
clusters = km_res.cluster_centers_

plt.scatter(x = data['V1'],y = data['V2'],c = km_res.labels_, alpha=0.3)
plt.scatter(clusters[:,0],clusters[:,1],color='red')

# Nomeando os eixos 'x' e 'y'
plt.xlabel('V1')
plt.ylabel('V2')
plt.title("Analyzing characteristics", 
          fontdict={'family': 'monospace', 
                    'color' : 'black',
                    'weight': 'bold',
                    'size': 16},
          loc='center')    

# Plotando os pontos num gráfico
plt.show()
