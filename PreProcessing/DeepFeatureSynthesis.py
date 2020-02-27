import featuretools as ft
import utils


class PreProcessor:

    def __init__(self, data):
        es = ft.EntitySet("datata")

        feature_matrix, features = ft.dfs(target_entity='data_set', entityset=es, verbose=True)

        fm_encoded, features_encoded = ft.encode_features(feature_matrix, features)

        print("Number of features %s" % len(features_encoded))
        fm_encoded.head(10)

        return
