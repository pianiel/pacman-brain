

def loadQTable(qfilename):
    qtable = {}
    try:
        with open(qfilename) as f:
            for line in f:
                key, val = line.split('\t')
                key = eval(key.strip())
                val = float(val.strip())
                qtable[key] = val
    except:
        print "Error while loading qtable from", qfilename, " - returning empty dict"
        return {}
    return qtable

# qtab = loadQTable('q.log')
# print qtab.items()