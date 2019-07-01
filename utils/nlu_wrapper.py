import os

import requests
from sutime import SUTime

class NLUWrapper(object):
    def __init__(self, host='localhost', port=5001, **kwargs):
        self.host, self.port = host, port
        self.sutime = SUTime(jars=os.path.join(os.path.dirname(__file__), 'python-sutime', 'jars'),
                             mark_time_ranges=True)
        print 'Initialized with {}:{}'.format(self.host, self.port)

    def annotate(self, in_utterance, modules=()):
        sutime_response = None
        try:
            if 'SUTime' in modules:
                sutime_response = self.sutime.parse(in_utterance)
                modules = [module for module in modules if module != 'SUTime']
            response = requests.post('http://{}:{}/annotate'.format(self.host, self.port),
                                     json={'state': {'utterance': in_utterance},
                                           'modules': modules},
                                     timeout=5)
        except requests.Timeout:
            return {}
        assert response.status_code == 200, 'Error calling the NLU service'
        result = response.json()
        if sutime_response is not None:
            result['annotations']['SUTime'] = sutime_response
        return result

    def annotate_sentiment(self, in_utterance):
        response = self.annotate(in_utterance, modules=['Preprocessor', 'VaderNLTK'])
        return response['annotations']['sentiment']

    def annotate_ner(self, in_utterance):
        response = self.annotate(in_utterance, modules=['Preprocessor', 'StanfordNER'])
        return response['annotations'].get('ner', {})

    def annotate_pos(self, in_utterance):
        response = self.annotate(in_utterance, modules=['Preprocessor', 'MorphoTagger'])
        return response['annotations'].get('postag', [])

    def annotate_abuse(self, in_utterance):
        response = self.annotate(in_utterance, modules=['Preprocessor', 'AlanaAbuseDetector'])
        return response['annotations'].get('abuse', {})

