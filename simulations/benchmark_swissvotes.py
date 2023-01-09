# import geopandas as gpd

# import pyproj
# from shapely.ops import transform

# import json
# import plotly.express as px

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import networkx as nx
from pyvis.network import Network

from StructuredGraphLearning.utils import Operators
from StructuredGraphLearning.metrics import Metrics

import networkx.algorithms.community as nx_comm

from src.estimators import SGLkComponents, EllipticalGL


# f=r"/home/AHippert-Ferrer/Documents/graphfactormodel/data/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp"

# shapes = gpd.read_file(f)
# shapes.geometry = shapes.geometry.simplify(0.5)

# gdf = gpd.GeoDataFrame(columns=shapes.columns)

# lv95 = pyproj.CRS('EPSG:2056')
# wgs84 = pyproj.CRS('EPSG:4326')
# project = pyproj.Transformer.from_crs(lv95, wgs84, always_xy=True).transform

# for i, shape in shapes.iterrows():
#     if shape.geometry is not None:
#         shape.geometry = transform(project, shape.geometry)
#         shapes.iloc[i] = shape
     
# for i, shape in shapes.iterrows():
#     if shape.geometry is not None:
#         if shape.NAME in gdf.NAME.values:
#             available_geometry = gdf.loc[gdf.NAME == shape.NAME,'geometry'].geometry.values
#             gdf.loc[gdf.NAME == shape.NAME, 'geometry']=available_geometry.union(shape.geometry)
#         else:
#             gdf = gdf.append(shape)
# gdf = gdf.drop(columns=['KT_TEIL'])


# gdf = gdf.rename(columns={'EINWOHNERZ':'Inhabitants'})

df = pd.read_csv("../data/votations.csv", dtype={"INDEX": str})

df['mean'] = df.mean(axis=1)
names_array = df['NAME']
names_array = np.hstack(np.hstack(names_array))
dict_names = {}
for i, name in enumerate(names_array):
    dict_names[i] = name

X = df.values[:, 2:] # Remove Id and Canton name to only keep voting percentages
X = X.astype(np.float32)
X = X.T

n_samples, n_features = X.shape
n_clusters = 3

cluster_colors = [
    'red', 'blue', 'black', 'orange',
    'green', 'gray', 'purple', 'cyan',
    'yellow', 'brown', 'magenta'
]

#X = df.values[:, 2:].to_numpy(dtype=np.float32)
print(X.shape)

# Pre-processing
pre_processing = StandardScaler(with_mean=True, with_std=False)

SGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            None, maxiter=1000, record_objective=True, record_weights=True,
            beta=1000, k=n_clusters, verbosity=1)),
        #('Clustering', louvain(modularity='Newman', n_aggregations=n_clusters))
        ]
    )

EllGL = Pipeline(steps=[
    ('Centering', pre_processing),
    ('Graph Estimation', EllipticalGL(geometry="factor+penalty", k=3,
                                      lambda_seq=[10], df=1e3)),
]
                 )

list_names = ['EllGL']
list_pipelines = [EllGL]

# Doing estimation
for pipeline, name in zip(list_pipelines, list_names):
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("Doing estimation with :", name)
    print(pipeline)
    pipeline.fit_transform(X)

    # Get adjacency matrix to get the graph and clustered labels to compute modularity
    adjacency = pipeline['Graph Estimation'].precision_
    #labels = pipeline['Clustering'].labels_
    graph = nx.from_numpy_matrix(adjacency)
    graph = nx.relabel_nodes(graph, dict_names)

    print('Graph statistics:')
    print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
    print('Modularity: ', nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph)))
    print("----------------------------------------------------------")
    print('\n\n')

    cluster_seq = nx_comm.label_propagation_communities(graph)
    print(cluster_seq)

    nt = Network(height='900px', width='100%')
    #nt.show_buttons()
    nt.set_options("""
    const options = { "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 1,
    "opacity": 1,
    "font": {
      "size": 15
    },
    "scaling": {
      "min": 17
    },
    "shadow": {
      "enabled": true
    },
    "size": 17
    },
    "edges": {
    "arrowStrikethrough": false,
    "color": {
      "inherit": true
    },
    "selfReferenceSize": null,
    "selfReference": {
      "angle": 0.7853981633974483
    },
    "shadow": {
      "enabled": true
    },
    "smooth": {
      "type": "continuous",
      "forceDirection": "none",
      "roundness": 1
    }
    },
    "physics": {
    "forceAtlas2Based": {
      "springLength": 100
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based"
  }
    }
    """)
        
    nt.from_nx(graph)

    # Adding labels if possible
    if len(cluster_seq) > 0:
        for i_color, i_cluster in enumerate(cluster_seq):
            for i_crypto in i_cluster:
                nt.get_node(i_crypto)['color'] = cluster_colors[i_color]

    nt.show(f'swiss_votes.html')

#plt.show()


# plot_clusters = False
# if plot_clusters:
#     conditions = [(df.Immigre_09_02_2014<50),
#                   (df.Immigre_09_02_2014>=50) & (df.Immigre_09_02_2014<58.15),
#                   (df.Immigre_09_02_2014)>=58.15]

#     values = ["Against", "Support", "Strong support"]

#     df['cluster'] = np.select(conditions, values)

#     fig = px.choropleth_mapbox(df, geojson=gdf.geometry,
#                                locations='INDEX',
#                                color='cluster',
#                                hover_name = 'NAME',
#                                #color_continuous_scale='algae',
#                                mapbox_style="carto-positron",
#                                zoom=7,
#                                center = {"lat": 46.8, "lon": 8.5},
#                                opacity=0.5,
#                                )
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#     fig.show()
