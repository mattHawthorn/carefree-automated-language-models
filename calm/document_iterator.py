import os

########################################
# DOCUMENT ITERATOR ####################
########################################
        

class DocIter:
    """
    Iterates over docs in a directory or a file.  recordIter is any iterable returning docs as dicts of attribute:value pairs.
    you can supply any class here that will iterate over docs/records in a file, given a path to the file. 
    By using this iterator, you never need to have more than the contents of one input file in memory at a time during processing/analysis.
    """
    
    def __init__(self,path,recordIter,recursive=False,extensions=None):
        # what kind of files to read?
        self.extensions = set(extensions)
        self.recursive = recursive
        self.recordIter = recordIter
        self.path = path
        
    def walk(self):
        path = self.path
        # if path exists,
        if os.path.exists(path):
            # and path is a directory
            if os.path.isdir(path):
                # set up a recursive path walk iterator
                dirIter = os.walk(path)
                # but if not recursive, only take the contents of path as dirIter
                if not self.recursive:
                    dirIter = iter([next(dirIter)])
                    
            # otherwise assume path is a file.
            else:
                # no iteration on directories or files; dirIter will just have one parentDir, subDir, and filename
                directory, filename = os.path.split(path)
                subDir = os.path.split(directory)[1]
                # structure these as in the output from os.walk()
                dirIter = iter([(directory,[subDir],[filename])])
                    
        # else, path doesn't exist
        else:
            raise FileNotFoundError(path+" not found. Iterator not initiated")
        
        return dirIter
    
    def __iter__(self):
        for directory,subdirs,files in self.walk():
            for path in [os.path.join(directory,f) for f in files if os.path.splitext(f)[-1] in self.extensions]:
                for doc in self.recordIter(path):
                    yield doc    
    

# Example record iterator: this one returns records from a .json of yelp reviews 
#class RecordIter:
#    # iterates over individual records in a file, in this case a json with records in a slot labelled 'Reviews'
#    def __init__(self, filepath, encodings=['iso-8859-1','utf-16le']):
#        with open(filepath,'r') as readfile:
#            data = json.load(readfile)
#            self.data = data['Reviews']
#            self.docs = iter(self.data)
#            
#    def __iter__(self):
#        return self
#    
#    def __next__(self):
#        try:
#            nextdoc = next(self.docs)
#        except:
#            raise StopIteration
#        else:
#            return nextdoc

