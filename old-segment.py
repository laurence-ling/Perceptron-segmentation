#!/usr/bin/python3

import random, pickle, os
import numpy as np

class Perceptron:
    def __init__(self):
        self.train_set = []
        self.feature_set = {}
        self.weight = []
        self.dimension = 0 #feature vector's dimension

    def train(self, weight_filename):
        self.weight_file = weight_filename
        #self.feature_set = pickle.load(open(feature_filename, 'rb'))
        self.dimension = len(self.feature_set)
        if os.path.exists(weight_filename):
            fw = open(weight_filename, 'rb')
            self.weight = pickle.load(fw)
            fw.close()
        else:
            self.weight = np.zeros(self.dimension)
        print(self.weight)  
        r = 0
        for sent in self.train_set[0:30000]:
            print(r)
            r += 1
            saved_weight = []
            for i in range(len(sent)):
                # argmax(y)
                predict_y = self.argmaxy(sent, i)           
                if predict_y != sent[i][1]:
                    # update weight
                    delta = self.get_feature(sent, i, sent[i][1]) - self.get_feature(sent, i, predict_y)
                    self.weight = self.weight + delta
                saved_weight.append(self.weight)
            # average weight
            self.weight = sum(saved_weight)/len(saved_weight)
        
        fd = open(weight_filename, 'wb')
        pickle.dump(self.weight, fd)
        fd.close()


    def argmaxy(self, sent, idx):
        labels = ['B', 'M', 'E', 'S']
        predict_y = random.choice(labels)
        maxval = -10000
        for y in labels:
            vector = self.get_feature(sent, idx, y)
            value = vector.dot(self.weight)
            print(y, value)
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
            features.append(sent[idx - 1][0] + char + '_' + label)
            features.append(sent[idx - 1][0] + '~' + label)
        
        vec = np.zeros(len(self.weight))
        for f in features:
            if f in self.feature_set:
                vec[self.feature_set[f]] = 1
        return vec
        
    def predict(self, testset_filename, feature_filename, weight_filename):
        if not self.weight:
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
                print('blank line')
                continue
            sent = [[char, ''] for char in sent]
            # predict label
            for i in range(len(sent)):
                sent[i][1] = self.argmaxy(sent, i)
            print(sent)
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
                    
            print(' '.join(segmented))
            return

    def extract(self, trainset_filename, feature_filename, tagged_filename):
        self.feature_file = feature_filename
        fd = open(trainset_filename, 'r')
        feature_set = set()
        for sent in fd.readlines():
            sent = sent.strip().split()
            if not sent:
                print('blank line',sent, len(self.train_set))
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
                #trigram
                if (i - 1 >= 0 and i + 1 < len(tagged)):
                    pass
        fd.close()
        feature_set = list(feature_set)
        for i in range(len(feature_set)):
            self.feature_set[feature_set[i]] = i
        fw = open(feature_filename, 'wb')
        pickle.dump(self.feature_set, fw)
        fw.close()
        '''
        if not os.path.exists(tagged_filename):
            fw = open(tagged_filename, 'wb')
            pickle.dump(self.train_set, fw)
            fw.close()
        '''
        print("feature extract complete", len(self.train_set))

def main():
    classifier = Perceptron()
    #classifier.extract('data/train.txt', 'feature_set', 'train_tagged')
    #classifier.train('weight')
    classifier.predict('data/test.txt', 'feature_set', 'weight')

if __name__ == '__main__':
    main()