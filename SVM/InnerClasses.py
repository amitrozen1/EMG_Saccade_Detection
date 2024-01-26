from dataclasses import dataclass
import json
from json import JSONEncoder

class SimpleEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

@dataclass
class SvmComponentWeight:
    component : int
    electrode : int
    relevance : float

    @staticmethod
    def JsonDumps(obj):
        return json.dumps(obj, cls=SimpleEncoder)

    def JsonLoads(dict):
        return SvmComponentWeight(dict["component"], dict["electrode"], dict["relevance"])

def LoadSvmComponentsArrFromFile(file_path):
    list_of_jsons = ""
    with open(file_path, "r") as f:
        list_of_jsons = json.load(f)
    return [SvmComponentWeight.JsonLoads(dict) for dict in list_of_jsons]

if __name__ == "__main__":
    a = SvmComponentWeight(1,1,2.3)
    b = [a,a,a]
    print(SvmComponentWeight.JsonDumps(b))