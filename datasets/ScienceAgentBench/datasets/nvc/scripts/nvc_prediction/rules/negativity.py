import ccobra

from . import atmosphere

def is_negative(quantifier):
    return quantifier in ['No', 'Some not']

class Negativity(ccobra.CCobraModel):
    def __init__(self, name='NegativityRule'):
        super(Negativity, self).__init__(
            name, ['syllogistic'], ['single-choice'])

        self.atmosphere = atmosphere.Atmosphere()

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)
        if is_negative(item.task[0][0]) and is_negative(item.task[1][0]):
            return syllogism.decode_response('NVC')

        return self.atmosphere.predict(item)

    def adapt(self, item, truth, **kwargs):
        pass
