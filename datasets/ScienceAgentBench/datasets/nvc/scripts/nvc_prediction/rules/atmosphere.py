import pandas as pd
import numpy as np

import ccobra

def atmosphere_predictions(premises):
    responses = []
    if premises == 'AA':
        responses = ['Aac', 'Aca']
    elif premises == 'AI':
        responses = ['Iac', 'Ica']
    elif premises == 'AE':
        responses = ['Eac', 'Eca']
    elif premises == 'AO':
        responses = ['Oac', 'Oca']
    elif premises == 'EE':
        responses = ['Eac', 'Eca']
    elif premises == 'EI':
        responses = ['Oac', 'Oca']
    elif premises == 'EO':
        responses = ['Oac', 'Oca']
    elif premises == 'II':
        responses = ['Iac', 'Ica']
    elif premises == 'IO':
        responses = ['Oac', 'Oca']
    elif premises == 'OO':
        responses = ['Oac', 'Oca']
    return responses

class Atmosphere(ccobra.CCobraModel):
    def __init__(self, name='Atmosphere'):
        super(Atmosphere, self).__init__(
            name, ['syllogistic'], ['single-choice'])

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)
        enc_resps = atmosphere_predictions(''.join(sorted(syllogism.encoded_task[:2])))
        return syllogism.decode_response(np.random.choice(enc_resps))
