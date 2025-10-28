import ccobra

from . import atmosphere

class PartNeg(ccobra.CCobraModel):
    def __init__(self, name='PartNeg'):
        super(PartNeg, self).__init__(
            name, ['syllogistic'], ['single-choice'])

        self.atmosphere = atmosphere.Atmosphere()

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)

        if item.task[0][0] != 'All' and item.task[1][0] != 'All':
            return syllogism.decode_response('NVC')

        return self.atmosphere.predict(item)

    def adapt(self, item, truth, **kwargs):
        pass
