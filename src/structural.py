import gensim.models.doc2vec as doc
import os
import random
import networkx as nx

def randomWalk(G, walkSize):
    walkList= []
    curNode = random.choice(G.nodes())

    while(len(walkList) < walkSize):
        walkList.append(G.node[curNode]['label'])
        curNode = random.choice(G.neighbors(curNode))
    return walkList

def getStats(G):
    stats ={}
    stats['num_nodes'] = nx.number_of_nodes(G)
    stats['num_edges'] = nx.number_of_edges(G)
    stats['is_Connected'] = nx.is_connected(G)


def drawGraph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("graph.pdf")
    plt.show()

def arr2str(arr):
    result = ""
    for i in arr:
        result += " "+str(i)
    return result


def generateDegreeWalk(Graph, walkSize):
    g = Graph
    walk = randomWalkDegreeLabels(g,walkSize)
    #walk = serializeEdge(g,NodeToLables)
    return walk

def randomWalkDegreeLabels(G, walkSize):
    curNode = random.choice(G.nodes())
    walkList= []

    while(len(walkList) < walkSize):
        walkList.append(G.node[curNode]['label'])
        curNode = random.choice(G.neighbors(curNode))
    return walkList

def getDegreeLabelledGraph(G, rangetoLabels):
    degreeDict = G.degree(G.nodes())
    labelDict = {}
    for node in degreeDict.keys():
        val = degreeDict[node]/float(nx.number_of_nodes(G))
        labelDict[node] = inRange(rangetoLabels, val)
        #val = degreeDict[node]/float(nx.number_of_nodes(G))
        #labelDict[node] = degreeDict[node]

        nx.set_node_attributes(G, 'label', labelDict)

    return G

def inRange(rangeDict, val):
        for key in rangeDict:
            if key[0] < val and key[1] >= val:
                return rangeDict[key]

def generateWalkFile(dirName, walkLength, alpha):
    walkFile = open(dirName+'.walk', 'w')
    indexToName = {}
    rangetoLabels = {(0, 0.05):'z',(0.05, 0.1):'a', (0.1, 0.15):'b', (0.15, 0.2):'c', (0.2, 0.25):'d', (0.25, 0.5):'e', (0.5, 0.75):'f',(0.75, 1.0):'g'}
    for  root, dirs, files in os.walk(dirName):
        index = 0
        for name in files:
            print(name)

            subgraph = graphUtils_s.getGraph(os.path.join(root, name))
            degreeGraph = getDegreeLabelledGraph(subgraph, rangetoLabels)
            degreeWalk = generateDegreeWalk(degreeGraph, int(walkLength* (1- alpha)))
            walk = graphUtils_s.randomWalk(subgraph, int(alpha * walkLength))
            walkFile.write(arr2str(walk)+ arr2str(degreeWalk) +"\n")
            indexToName[index] = name
            index += 1
    walkFile.close()

    return indexToName

def saveVectors(vectors, outputfile, IdToName):
    output = open(outputfile, 'w')

    output.write(str(len(vectors)) +"\n")
    for i in range(len(vectors)):
        output.write(str(IdToName[i]))
        for j in vectors[i]:
            output.write('\t'+ str(j))
        output.write('\n')
    output.close()


def structural_embedding(args):

    inputDir = args.input
    outputFile = args.output
    iterations = args.iter
    dimensions = args.d
    window = args.windowSize
    dm = 1 if args.model == 'dm' else 0
    indexToName = generateWalkFile(inputDir, args.walkLength, args.p)
    sentences = doc.TaggedLineDocument(inputDir+'.walk')

    model = doc.Doc2Vec(sentences, size = dimensions, iter = iterations, dm = dm, window = window )

    saveVectors(list(model.docvecs), outputFile, indexToName)
