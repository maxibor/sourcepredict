import sys
import os
parentScriptDir = "/".join(os.path.dirname(
    os.path.realpath(__file__)).split("/")[:-1])
sys.path.append(parentScriptDir+"/sourcepredictlib")

print(sys.path)

import ml
