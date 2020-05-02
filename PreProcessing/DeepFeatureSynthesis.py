import featuretools as ft
import utils


class PreProcessor:

    def __init__(self, data):
        es = ft.EntitySet("transactions")

        es = es.entity_from_dataframe(entity_id='entities_transactions', dataframe=data, index='index_col')

        es.normalize_entity(base_entity_id='entities_transactions', new_entity_id='origin', index='type')

        fm, features = ft.dfs(entityset=es, target_entity='entities_transactions')

        self.feature_matrix = fm
        self.features = features

        return
