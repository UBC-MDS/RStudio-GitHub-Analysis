"""
To get the embeddings for the motifs so that we can group them based on their similarity,
and further cluster them to find the common sub-patterns of the graphs

Input: pickle, saved motifs as networkx graphs
output: embeddings saved in a file
"""

import gensim.models.doc2vec as doc
import os
import random
import networkx as nx
import pickle

class Sub2Vec:

    def __init__(self, dimensions=128, iterations=20, window=2):
        self.size = dimensions
        self.iter = iterations
        self.window = window

    # Generate the random walk
    def generateDegreeWalk(self, G, walkSize):
        """
        The function generates a random walk from a random sampled node in the graph.
        Parameters
        ----------
            G: networkx graph, the degree-labelled motifs;
            walkSize: length of walk
        Returns
        -------
        a list, record the random walk generated, noted by the degree of each node
        """
        curNode = random.choice(G.nodes())
        walkList= []

        while(len(walkList) < walkSize):
            walkList.append(G.node[curNode]['label'])
            curNode = random.choice(G.neighbors(curNode))
        return walkList

    def getDegreeLabelledGraph(self, G, rangetoLabels):
        """
        The function generates a new graph from the existing graph, with each node labelled by its degree.
        Parameters
        ----------
            G: networkx graph, the motifs;
            rangetoLabels: a range for the nodes to label, rather than the precise degree
        Returns
        -------
        networkx graph, relabelled with degree
        """
        degreeDict = G.degree(G.nodes())
        labelDict = {}
        for node in degreeDict.keys():
            val = degreeDict[node]/float(nx.number_of_nodes(G))
            labelDict[node] = inRange(rangetoLabels, val)

        nx.set_node_attributes(G, 'label', labelDict)

        return G


    def generateWalkFile(self, inputFile, walkFileName):
        """
        The function reads in a file with a set of motifs, outputs a file with random walk generated.
        Parameters
        ----------
            inputFileName: a pickle file of multiple pickles, each saved a motif as networkx graphs,
            walkFileName: name of walk file for output
        Returns
        -------
        a walk file, recorded all the walks
        """
        walkFile = open(walkFileName, 'w')
        indexToName = {}
        # range can be updated if we find more proper values
        rangetoLabels = {(0, 0.05):'z',(0.05, 0.1):'a', (0.1, 0.15):'b', (0.15, 0.2):'c', (0.2, 0.25):'d', (0.25, 0.5):'e', (0.5, 0.75):'f',(0.75, 1.0):'g'}
        pickles = loadall(inputFile)
        index = 0
        for pickle in pickles:
            subgraph = read_gpickle(pickle)
            degreeGraph = self.getDegreeLabelledGraph(subgraph, rangetoLabels)
            degreeWalk = self.generateDegreeWalk(degreeGraph, walkSize)
            walkFile.write(arr2str(degreeWalk) +"\n")
            indexToName[index] = index+1
            index += 1
        walkFile.close()

        return indexToName


    def structural_embedding(self, inputFile, outputFile):
        indexToName = self.generateWalkFile(inputFile, args.walkLength)
        sentences = doc.TaggedLineDocument(inputFile+'.walk')
        self.model = doc.Doc2Vec(sentences, size = dimensions, iter = iterations, window = window )

        saveVectors(list(self.docvecs), outputFile, indexToName)

    # Helper functions - 1
    def arr2str(arr):
        """
        Input: a list
        Output: a string, turned from the input list, with space between each element
        """
        result = ""
        for i in arr:
            result += " "+str(i)
        return result

    # Helper functions - 2
    def inRange(rangeDict, val):
        """
        Input: a list
        Output: a string, turned from the input list, with space between each element
        """
            for key in rangeDict:
                if key[0] < val and key[1] >= val:
                    return rangeDict[key]

    # Helper functions - 3
    def loadall(filename):
        """
        To load all the pickles contained in one pickle file
        """
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    # Helper functions - 4
    def saveVectors(vectors, outputfile, ID):
        """
        Input: vectors, path of output file, motif ID
        Output: saves down the motif ID and its embeddings in the output file
        """
        output = open(outputfile, 'w')

        output.write(str(len(vectors)) +"\n")
        for i in range(len(vectors)):
            output.write(str(ID[i]))
            for j in vectors[i]:
                output.write('\t'+ str(j))
            output.write('\n')
        output.close()
