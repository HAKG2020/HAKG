
import random

from data_preprocess import *
def loadAllSubPathsByTyplesRemoveRepeatPaths(subpaths_file):
    map={}
    with open(subpaths_file) as f:
        for l in f:
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1]
            sentence=[y for y in splitByTab[2].split()[:]]
            #if len(sentence)>maxlen:
                #continue
            #if key not in tuples:
                #continue
            if key in map:
                map[key].add(splitByTab[2])
            else:
                tmp=set()
                tmp.add(splitByTab[2])
                map[key]=tmp
    result={}
    for key in map:
        result[key]=[]
        for path in map[key]:
            result[key].append([y for y in path.split()[:]])
    return result

def generateSubgraphsByAllSubpathsDirectlyAndSave(tuples, subpathsMap, DAGSaveFile):
    """
    generate DAGs by subpaths
    """
    output = open(DAGSaveFile, 'w')
    for tuple in tuples:
        arr=tuple.strip().split('-')
        start=arr[0]
        end= arr[1]
        if tuple not in subpathsMap:
            continue
        subpaths=subpathsMap[tuple]
        for i in range(1):
            map={}
            mapCheck={}
            if subpaths[0][0]=='None':
                continue
            for j in range(len(subpaths)):
                subpath=subpaths[j]
                for x in range(len(subpath)-1):
                    if subpath[x] in map:
                        if subpath[x+1] not in mapCheck[subpath[x]]:
                            map[subpath[x]].append(subpath[x+1])
                            mapCheck[subpath[x]].add(subpath[x+1])
                    else:
                        map[subpath[x]]=[subpath[x+1]]
                        mapCheck[subpath[x]]=set([subpath[x+1]])
            dependency, sequence, nodesLevel=subgraphToOrderedSequence(map, start, end)

            s=str(start)+'-'+str(end)+'#'
            for depend in dependency:
                s+=str(depend[0])+'-'+str(depend[1])+'\t'
            s+='#'
            for id in sequence:
                s+=str(id)+'\t'
            s+='#'
            for id in sequence:
                s+=str(id)+'-'+str(nodesLevel[id])+'\t'
            s+='\n'
            output.write(s)
            output.flush()

    output.close()
    output=None

def subgraphToOrderedSequence(edges, start, end):
    """
    set DAG to a topology ordered sequence
    """
    nodesLevel={}
    nodesSeq={}


    for key,values in edges.items():
        if key not in nodesLevel:
            nodesLevel[key]=-1
    queue=[]
    now=start
    queue.append(now)
    nodesLevel[now]=0
    nodesSeq[now]=len(nodesSeq)
    results=[]
    endNodeLevel=-1
    while len(queue)>0:
        now=queue.pop(0)
        children=edges[now]
        for node in children:
            if node==end:
                results.append([now,node])
                if endNodeLevel==-1:
                    endNodeLevel=nodesLevel[now]+1
            elif nodesLevel[node]==-1:
                queue.append(node)
                nodesLevel[node]=nodesLevel[now]+1
                nodesSeq[node]=len(nodesSeq)
                results.append([now,node])
            elif nodesSeq[node]>nodesSeq[now]:
                results.append([now,node])
    nodesSeq[end]=len(nodesSeq)
    items=nodesSeq.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    sequence=[ backitems[i][1] for i in range(len(items))]
    nodesLevel[end]=endNodeLevel
    return results, sequence, nodesLevel

if __name__ =='__main__':

    subpathsFile = 'ml/test_negative_path_100_new.txt'
    dependencySaveFile = 'ml/dag_test_negative_100.txt'

    subpathsMap = loadAllSubPathsByTyplesRemoveRepeatPaths(subpathsFile)

    tuples = list(subpathsMap.keys())
    print(tuples)
    generateSubgraphsByAllSubpathsDirectlyAndSave(tuples, subpathsMap, dependencySaveFile)
