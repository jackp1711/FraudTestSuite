import matplotlib.pyplot as plot
import numpy as np


class Plot:

    pre_results = []
    post_results = []
    graph_labels = []

    graph = None

    def __init__(self, pre_results, post_results, labels):
        self.pre_results = pre_results
        self.post_results = post_results
        self.graph_labels = labels

    def plot_bar_chart_comparison_acc_score(self, title):
        x = np.arange(len(self.graph_labels))

        width = 0.35

        fig, self.graph = plot.subplots()
        rects1 = self.graph.bar(x - width / 2, self.pre_results, width, label='Without feature engineering')
        # rects2 = self.graph.bar(x + width / 2, self.post_results, width, label='With feature engineering')

        self.graph.set_ylabel('Accuracy score')
        self.graph.set_title(title)
        self.graph.set_xticks(x)
        self.graph.set_xticklabels(self.graph_labels)
        self.graph.legend()

        self.auto_label(rects1)
        # self.auto_label(rects2)

        fig.tight_layout()
        plot.show()

    def plot_bar_chart_comparison_false(self, correct_preds, title):
        x = np.arange(len(self.graph_labels))

        width = 0.35

        fig, self.graph = plot.subplots()
        rects1 = self.graph.bar(x - width/2, self.pre_results, width=width, label='False Positives')
        rects2 = self.graph.bar(x + width/2, self.post_results, width=width, label='False Negatives')
        # rects3 = self.graph.bar(x + width*2, correct_preds, width, label='Correct predictions for fraud')

        self.graph.set_ylabel('Number of occurrences')
        self.graph.set_title(title)
        self.graph.set_xticks(x)
        self.graph.set_xticklabels(self.graph_labels)
        self.graph.legend()

        self.auto_label(rects1)
        self.auto_label(rects2)
        # self.auto_label(rects3)

        # fig.tight_layout()
        plot.show()

    def auto_label(self, reacts):
        for rect in reacts:
            height = rect.get_height()
            self.graph.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0,3),
                                 textcoords="offset points", ha='center', va='bottom')