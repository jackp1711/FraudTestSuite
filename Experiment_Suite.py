import pandas as pd
import data_analysis as da


chunk_list = []

chunked_data = pd.read_csv('C:\\Users\\jacky\\PycharmProjects\\FraudTestSuite\\data\\simulated_data.csv',
                           chunksize=100000, low_memory=False, dtype={'step': 'int', 'type': object, 'amount': 'float',
                            'nameOrig': object, 'oldBalanceOrig': 'float', 'newBalanceOrig': 'float', 'nameDest': object
                            , 'oldBalanceDest': 'float', 'newBalanceDest': 'float', 'isFraud': 'int', 'isFlaggedFraud':
                            'int'})

for chunk in chunked_data:
    chunk_list.append(chunk)

simulated_data = pd.concat(chunk_list)
# new_list = [chunk_list[0], chunk_list[1], chunk_list[2], chunk_list[3], chunk_list[4], chunk_list[5]]
# simulated_data = pd.concat(new_list)

cc_data = pd.read_csv('C:\\Users\\jacky\\PycharmProjects\\FraudTestSuite\\data\\creditcard.csv')

simulated_data['index_col'] = simulated_data.index

sim_analysis_none = da.AnalysisTool(simulated_data, 'sim', 'isFraud')
sim_analysis = da.AnalysisTool(simulated_data, 'sim', 'isFraud')
# cc_analysis = da.AnalysisTool(cc_data, 'cc', 'Class')

# sim_analysis_none.run_experiments(5, 0.2, 101, 0)
# sim_analysis_none.graph_results("Simulated data ")
# del sim_analysis_none

sim_analysis.run_experiments(5, 0.2, 101, 1)
sim_analysis.graph_results("Simulated data ")
del sim_analysis

# cc_analysis_none = da.AnalysisTool(cc_data, 'cc', 'Class')
# cc_analysis_none.run_experiments(5, 0.2, 101, 0)
# cc_analysis_none.graph_results("Credit card fraud data ")
# del cc_analysis_none

# sim_analysis.run_experiments(3, 0.3, 101, 1)
# sim_analysis.graph_results()
