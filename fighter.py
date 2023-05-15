# fighter is used to encode fighter information compactly
class fighter:

    def __init__(self, name, record):
        self.name = name
        self.record = record
        self.equilibriumVal = None

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    # the below is necessary to enable sorting compatibility
    def __eq__(self, other):
        return self.equilibriumVal == other.equilibriumVal

    def __ne__(self, other):
        return self.equilibriumVal != other.equilibriumVal

    def __lt__(self, other):
        return self.equilibriumVal < other.equilibriumVal

    def __gt__(self, other):
        return self.equilibriumVal > other.equilibriumVal

    def __le__(self, other):
        return self.equilibriumVal <= other.equilibriumVal

    def __ge__(self, other):
        return self.equilibriumVal >= other.equilibriumVal