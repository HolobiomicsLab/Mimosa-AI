import ccobra

from . import atmosphere

def is_particular(quantifier):
    return quantifier in ['Some', 'Some not']

class Particularity(ccobra.CCobraModel):
    def __init__(self, name='ParticularityRule'):
        super(Particularity, self).__init__(
            name, ['syllogistic'], ['single-choice'])

        self.atmosphere = atmosphere.Atmosphere()

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)

        if is_particular(item.task[0][0]) and is_particular(item.task[1][0]):
            return syllogism.decode_response('NVC')

        return self.atmosphere.predict(item)

    def adapt(self, item, truth, **kwargs):
        pass
