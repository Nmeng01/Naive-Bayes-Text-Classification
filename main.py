import sys
import string
import math
import copy

stopWords = [
    'about', 'all', 'along', 'also', 'although', 'among', 'and', 'any', 'anyone', 'anything',
    'are', 'around', 'because', 'been', 'before', 'being', 'both', 'but', 'came', 'come',
    'coming', 'could', 'did', 'each', 'else', 'every', 'for', 'from', 'get', 'getting',
    'going', 'got', 'gotten', 'had', 'has', 'have', 'having', 'her', 'here', 'hers', 'him',
    'his', 'how', 'however', 'into', 'its', 'like', 'may', 'most', 'next', 'now', 'only',
    'our', 'out', 'particular', 'same', 'she', 'should', 'some', 'take', 'taken', 'taking',
    'than', 'that', 'the', 'then', 'there', 'these', 'they', 'this', 'those', 'throughout',
    'too', 'took', 'very', 'was', 'went', 'what', 'when', 'which', 'while', 'who', 'why',
    'will', 'with', 'without', 'would', 'yes', 'yet', 'you', 'your', 'com', 'doc', 'edu',
    'encyclopedia', 'fact', 'facts', 'free', 'home', 'htm', 'html', 'http', 'information',
    'internet', 'net', 'new', 'news', 'official', 'page', 'pages', 'resource', 'resources',
    'pdf', 'site', 'sites', 'usa', 'web', 'wikipedia', 'www', 'one', 'ones', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'tens', 'eleven', 'twelve',
    'dozen', 'dozens', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
    'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
    'hundred', 'hundreds', 'thousand', 'thousands', 'million', 'millions'
]

correct = 0
totalPreds = 0

# Calculate the negative log properties
def logProbs(catBioOcc, catWordOcc, corpusSize, epsilon):
    logLplcnCat = {}
    for c in catBioOcc:
        catFreq = catBioOcc[c]/corpusSize
        lplcnCat = (catFreq + epsilon)/(1 + (len(catBioOcc) * epsilon))
        logLplcnCat[c] = -math.log(lplcnCat, 2)
    logLplcnWord = {}
    for w, cats in catWordOcc.items():
        logLplcnWord[w] = {}
        for c in catBioOcc:
            condWordFreq = cats[c]/catBioOcc[c] if c in cats else 0
            lplcnWord = (condWordFreq + epsilon)/(1 + (2 * epsilon))
            logLplcnWord[w][c] = -math.log(lplcnWord, 2)
    
    return logLplcnCat, logLplcnWord

def predict(currName, currCat, catScores):
    global correct, totalPreds
    totalPreds += 1
    predCat = ''
    m = float('inf')
    # Get predicted category and min score m
    for c, score in catScores.items():
        if score < m:
            predCat = c
            m = score
    if predCat == currCat: correct += 1
    print(f'{currName}. Prediction: {predCat}. {"Right" if predCat == currCat else "Wrong"}')
    # Calculate actual probabilities
    xVals = {}
    for c, ci in catScores.items():
        if ci - m < 7:
            xVals[c] = 2**(m - ci)
        else:
            xVals[c] = 0
    s = sum(xVals.values())
    probStr = ''
    for c, x in xVals.items():
        probStr += f'{c}: {x/s:.2f}   '
    print(probStr)


def main():
    global correct, totalPreds
    catBioOcc = {}
    catWordOcc = {}
    if len(sys.argv) != 3:
        print("Please use a command in this format: python3 main.py corpus.txt 5")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as corpus:
        lineNum = 0
        currCat = ''
        currName = ''
        bioCnt = 0
        wordSet = set()
        testFlag = 0
        catScores = {}
        logLplcnCat = {}
        logLplcnWord = {}
        for line in corpus:
            # Remove whitespace and punctuation
            line = line.strip().translate(str.maketrans('', '', string.punctuation))
            # Move to next bio
            if not line:
                if lineNum != 0:
                    bioCnt += 1
                    if bioCnt == int(sys.argv[2]) and testFlag == 0:
                        logLplcnCat, logLplcnWord = logProbs(catBioOcc, catWordOcc, int(sys.argv[2]), 0.1)
                        testFlag = 1
                    elif testFlag == 1:
                        predict(currName, currCat, catScores)
                # Reset for next bio
                lineNum = 0
                currCat = ''
                currName = ''
                wordSet = set()
                if testFlag == 1:
                    catScores = copy.copy(logLplcnCat)
                continue
            else:
                # Do not include name or category in training or evaluating
                if lineNum == 0:
                    currName = line
                    lineNum += 1
                    continue
                elif lineNum == 1:
                    # Count the category when training
                    if testFlag == 0:
                        catBioOcc[line] = catBioOcc[line] + 1 if line in catBioOcc else 1
                    currCat = line
                    lineNum += 1
                    continue
                # Categorize each word
                line = line.lower()
                for w in line.split():
                    if w in  stopWords or w in wordSet:
                        continue
                    wordSet.add(w)
                    # Categorize each word when training
                    if testFlag == 0:
                        if w not in catWordOcc:
                            catWordOcc[w] = {}
                        if currCat in catWordOcc[w]:
                            catWordOcc[w][currCat] = catWordOcc[w][currCat] + 1 
                        else:
                            catWordOcc[w][currCat] = 1
                    # Calculate the score for each category
                    else:
                        if w not in catWordOcc:
                            continue
                        for c in catBioOcc:
                            catScores[c] += logLplcnWord[w][c]
                lineNum += 1
    # One more predict to cover the last bio
    predict(currName, currCat, catScores)
    print(f'Overall Accuracy: {correct} out of {totalPreds} = {correct/totalPreds:.2f}')

if __name__ == "__main__":
    main()
