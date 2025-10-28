import ccobra

from . import atmosphere

class Figural(ccobra.CCobraModel):
    def __init__(self, name='FiguralRule'):
        super(Figural, self).__init__(
            name, ['syllogistic'], ['single-choice'])

        self.atmosphere = atmosphere.Atmosphere()

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)

        if syllogism.figure in [3, 4]:
            return syllogism.decode_response('NVC')

        return self.atmosphere.predict(item)

    def adapt(self, item, truth, **kwargs):
        pass
