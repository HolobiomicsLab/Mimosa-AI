import ccobra

from . import atmosphere

def is_negative(quantifier):
    return quantifier in ['No', 'Some not']

def is_particular(quantifier):
    return quantifier in ['Some', 'Some not']

class EmptyStart(ccobra.CCobraModel):
    def __init__(self, name='EmptyStart'):
        super(EmptyStart, self).__init__(
            name, ['syllogistic'], ['single-choice'])

        self.atmosphere = atmosphere.Atmosphere()

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)

        if syllogism.figure == 1 and is_negative(item.task[0][0]) and not is_particular(item.task[0][0]):
            return syllogism.decode_response('NVC')
        if syllogism.figure == 2 and is_negative(item.task[1][0]) and not is_particular(item.task[1][0]):
            return syllogism.decode_response('NVC')

        return self.atmosphere.predict(item)

    def adapt(self, item, truth, **kwargs):
        pass
