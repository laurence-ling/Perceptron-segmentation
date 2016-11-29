#!/usr/bin/python3

import random, pickle, os, sys
import numpy as np

class Perceptron:
    def __init__(self):
        self.train_set = []
        self.feature_set = {}
        self.weight = []
        self.dimension = 0 #feature vector's dimension

    def train(self, feature_filename, weight_filename):
        if self.feature_set == {}:
            fw = open(feature_filename, 'rb')
            self.feature_set = pickle.load(fw)
            fw.close()
        self.dimension = len(self.feature_set)
        if os.path.exists(weight_filename):
            fw = open(weight_filename, 'rb')
            self.weight = pickle.load(fw)
            fw.close()
        else:
            self.weight = np.zeros(self.dimension)

        saved_weight = np.zeros(self.dimension)
        times = 0
        for sent in self.train_set:
            #print(times) 
            for i in range(len(sent)):
                # argmax(y)
                predict_y = self.argmaxy(sent, i)           
                if predict_y != sent[i][1]:
                    # update weight
                    vec1 = self.get_feature(sent, i, sent[i][1]) 
                    vec2 = self.get_feature(sent, i, predict_y)
                    for i in vec1:
                        self.weight[i] += 1
                    for j in vec2:
                        self.weight[j] -= 1
            saved_weight += self.weight
            times += 1
        # average weight
        print('times', times)
        
        self.weight = saved_weight / times
        fd = open(weight_filename, 'wb')
        pickle.dump(self.weight, fd)
        fd.close()

    def argmaxy(self, sent, idx):
        labels = ['B', 'M', 'E', 'S']
        predict_y = random.choice(labels)
        maxval = -10000
        for y in labels:
            vec = self.get_feature(sent, idx, y)
            value = 0
            for i in vec:
                value += self.weight[i]
            if value > maxval: 
                maxval = value
                predict_y = y
        #print(predict_y)
        return predict_y

    def get_feature(self, sent, idx, label):
        char = sent[idx][0]
        features = [char + '_' + label]
        if (idx + 1 < len(sent)):
            features.append(char + '_' + label + sent[idx + 1][0])
            features.append(label + '~' + sent[idx + 1][0])
        if (idx - 1 >= 0):
            features.append(sent[idx-1][0] + char + '_' + label)
            features.append(sent[idx-1][0] + '~' + label)    
            #features.append(sent[idx-1][1] + '~' + label)
        if (idx - 1 >= 0 and idx + 1 < len(sent)):
            features.append(sent[idx-1][0] + '~' + label + '~' + sent[idx+1][0])
        vec = []
        for f in features:
            if f in self.feature_set:
                vec.append(self.feature_set[f])
        return vec
        
    def predict(self, testset_filename, feature_filename, weight_filename):
        if self.weight == []:
            fw = open(weight_filename, 'rb')
            self.weight = pickle.load(fw)
            fw.close()
        if not self.feature_set:
            fs = open(feature_filename, 'rb')
            self.feature_set = pickle.load(fs)
            fs.close()

        fd = open(testset_filename, 'r')
        for sent in fd.readlines():
            sent = sent.strip()
            if not sent:
                continue
            sent = [[char, ''] for char in sent]
            # predict label
            for i in range(len(sent)):
                sent[i][1] = self.argmaxy(sent, i)
            #print(sent)
            # segment words
            segmented = []
            current = 0
            while current < len(sent):
                item = sent[current]
                # due to the miss tag, B E E may appear
                if item[1] == 'S' or item[1] == 'E':
                    segmented.append(item[0])
                    current += 1
                    continue
                elif item[1] == 'B' or item[1] == 'M':
                    temp = item[0]
                    j = current + 1
                    while j < len(sent):
                        if sent[j][1] == 'M':
                            temp += sent[j][0]
                            j += 1
                        elif sent[j][1] == 'E':
                            temp += sent[j][0]
                            j += 1
                            break
                        else:
                            break
                    segmented.append(temp)
                    current = j
            if len(sys.argv) == 2:     
                print(' '.join(segmented), sent)
            else:
                print(' '.join(segmented))

    def preprocess(self, trainset_filename):
        fd = open(trainset_filename, 'r')
        for sent in fd.readlines():
            sent = sent.strip().split()
            if not sent:
                #print('blank line',sent, len(self.train_set))
                continue
            tagged = []
            for words in sent:
                if len(words) == 1:
                    tagged.append((words, 'S'))
                    continue
                tagged.append((words[0], 'B'))
                for char in words[1:-1]:
                    tagged.append((char, 'M'))
                tagged.append((words[-1], 'E'))
            self.train_set.append(tagged)
        fd.close()
        if len(sys.argv) == 2:
            print('train set is ready', len(self.train_set))

    def extract(self, feature_filename):
        feature_set = set()
        for tagged in self.train_set:
            for i in range(len(tagged)):
                #unigram
                feature_set.add(tagged[i][0] + '_' + tagged[i][1])
                # bigrams
                if (i + 1 < len(tagged)):
                    feature_set.add(tagged[i][0] + '_' + tagged[i][1] + tagged[i+1][0])
                    feature_set.add(tagged[i][1] + '~' + tagged[i+1][0])
                if (i - 1 >= 0):
                    feature_set.add(tagged[i-1][0] + tagged[i][0] + '_' + tagged[i][1])
                    feature_set.add(tagged[i-1][0] + '~' + tagged[i][1])
                    #feature_set.add(tagged[i-1][1] + '~' + tagged[i][1])
                #trigram
                if (i - 1 >= 0 and i + 1 < len(tagged)):
                    feature_set.add(tagged[i-1][0] + '~' + tagged[i][1] + '~' + tagged[i+1][0])
                    
        
        feature_set = list(feature_set)
        for i in range(len(feature_set)):
            self.feature_set[feature_set[i]] = i

        fw = open(feature_filename, 'wb')
        pickle.dump(self.feature_set, fw)
        fw.close()
    
        print("feature extract complete", len(self.feature_set))

def main():
    classifier = Perceptron()
    classifier.preprocess('data/train.txt')
    #classifier.extract('feature_set')
    #classifier.train('feature_set', 'weight')
    classifier.predict('data/test.txt', 'feature_set', 'weight')

if __name__ == '__main__':
    main()