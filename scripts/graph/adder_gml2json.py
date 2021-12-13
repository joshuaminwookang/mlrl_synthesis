import os

## quick script to convert adder_gmals to adder_jsons
for i in range(210):
    command = 'python gml2json.py' + ' --input adder_gmls/adder_' + str(i) + '.gml --output adder_jsons/adder_' + str(i) + '.json'
    os.system(command)
    print(command)


