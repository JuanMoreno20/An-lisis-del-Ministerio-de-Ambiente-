from fastapi import FastAPI
import pandas as pd
from sklearn.cluster import KMeans

app = FastAPI()

#base de datos
Base_Cluster_Est = pd.read_csv("../../Datos/Raw/BaseDatos_Cluster.csv")
identificadores = pd.read_csv("../../Datos/Raw/identificadores.csv")
Base_filtrada = pd.read_csv("../../Datos/Raw/Base_filtrada.csv")

if 'Unnamed: 0' in Base_Cluster_Est.columns:
    Base_Cluster_Est.drop(columns=['Unnamed: 0'], inplace=True)

#cluster 
numero_clusters = 6
kmeans = KMeans(n_clusters=numero_clusters, random_state=42)
clusters = kmeans.fit_predict(Base_Cluster_Est)

Base_Cluster_Est['cluster'] = clusters

#medias y eso 
cluster_means = Base_Cluster_Est.groupby('cluster').mean().round(2)
centroids = pd.DataFrame(kmeans.cluster_centers_).round(2)

#merge
Base_Cluster_fusionada = Base_Cluster_Est.merge(Base_filtrada, left_index=True, right_index=True, how='right')

@app.get("/")
async def root():
    return {
        "message": "¡Bienvenido! Aquí tienes una lista de funciones disponibles y los comandos para invocarlas:",
        "functions": {
            "Obtener información de un contrato por índice": ".../indice/{index}",
            "Obtener información de un cluster por número de cluster": ".../cluster_info/{cluster}",
            "Obtener el número de fila de un contrato por id_contrato": ".../fila/{id_contrato}",
            "Comparar las características de los clusters": ".../comparar_clusters"
        }
    }

@app.get("/indice/{index}")
async def get_contract_info(index: int):
    try:
        
        contrato = Base_Cluster_fusionada.loc[index]
        
        cluster = int(contrato['cluster'])
        
        
        contrato_filtrado = contrato[Base_filtrada.columns]
        
        
        response = {
            "Índice": index,
            "Contrato": contrato_filtrado.to_dict(),
            "Cluster": {
                "Número": cluster,
                "Descripción": "El contrato pertenece al cluster número " + str(cluster)
            }
        }
        return response
    except KeyError:
        return {"error": "Índice de contrato no encontrado"}

@app.get("/cluster_info/{cluster}")
async def get_cluster_info(cluster: int):
    try:
        
        column_names = Base_Cluster_Est.drop(columns=['cluster']).columns.tolist()
        
        
        cluster_info = {
            "media_del_cluster": cluster_means.loc[cluster].to_dict(),
            "centroides": dict(zip(column_names, centroids[cluster].round(2).tolist())),
            }
        
        
        response = {
            "Cluster": {
                "Número": cluster,
                "Información del Cluster": cluster_info
            }
        }
        return response
    except KeyError:
        return {"error": "Cluster no encontrado"}

    
@app.get("/fila/{id_contrato}")
async def get_numero_fila(id_contrato: int):
    try:
        
        numero_fila = int(identificadores[identificadores['id_contrato'] == id_contrato]['numero_fila'])
        
        response = {
            "ID": id_contrato,
            "Número de fila": numero_fila,
            "Nota": f"Para encontrar el cluster correspondiente, escribe /indice/{numero_fila}"
        }
        return response
    except KeyError:
        return {"error": "ID de contrato no encontrado"}
    except ValueError:
        return {"error": "ID de contrato no encontrado"}

@app.get("/comparar_clusters")
async def compare_clusters():
    try:
        
        cluster_info = pd.DataFrame()

        
        for i in range(numero_clusters):
            cluster_info[f"Cluster {i}"] = cluster_means.loc[i]

        
        cluster_info = cluster_info.T

        
        cluster_info_dict = cluster_info.to_dict()

        
        response = {
            "Comparación de Clusters": cluster_info_dict
        }
        return response
    except KeyError:
        return {"error": "No se pudo realizar la comparación de clusters"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run( "APPI:app", host="127.0.0.1", port=8000) 