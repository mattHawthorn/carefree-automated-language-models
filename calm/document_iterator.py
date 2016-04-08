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
        self.extensions = extensions
        self.recordIter = recordIter
        
        # if path exists,
        if os.path.exists(path):
            # and path is a directory
            if os.path.isdir(path):
                # set up a recursive path walk iterator
                self.dirIter = os.walk(path)
                # but if not recursive, only take the contents of path as dirIter
                if not recursive:
                    self.dirIter = iter([next(self.dirIter)])
                    
            # otherwise assume path is a file.
            else:
                # no iteration on directories or files; dirIter will just have one parentDir, subDir, and filename
                directory, filename = os.path.split(path)
                subDir = os.path.split(directory)[1]
                # structure these as in the output from os.walk()
                self.dirIter = iter([(directory,[subDir],[filename])])
                    
        # else, path doesn't exist
        else:
            print(path+" not found. Iterator not initiated")
        
        # start with file and doc/record iterators empty; these will be initialized from dirIter
        self.fileIter = iter([])
        self.docIter = iter([])
        
    
    def __iter__(self):
        return(self)
    
    
    def __next__(self):
        
        while True:
            # try to get a doc
            try:
                nextDoc = next(self.docIter)
            # if not, docIter is exhausted or uninitiated; get new file from fileIter
            except:
                try:
                    nextFile = next(self.fileIter)

                    # if not, fileIter is exhausted or uninitiated; get new dir from dirIter
                except:
                    try:
                        nextDir = next(self.dirIter)
                    # if not, dirIter is exhausted; end
                    except:
                        raise StopIteration
                    # if that worked, make a fileIter from nextDir
                    else:
                        self.fileIter = FileIter(directory=nextDir[0], files=nextDir[2], extensions=self.extensions)
                # if that worked, make a docIter from nextFile
                else:
                    self.docIter = self.recordIter(filepath=nextFile)
            # if so, return the doc
            else:
                return nextDoc
                break


class FileIter:
    # Iterates over files in directory, returning complete paths, as long as the file extension is in extensions
    def __init__(self, directory, files, extensions):
        self.directory = directory
        files = [filename for filename in files if os.path.splitext(filename)[1] in extensions]
        self.files = iter(files)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            nextfile = next(self.files)
        except StopIteration:
            raise StopIteration
        else:
            path = os.path.join(self.directory,nextfile)
            return path
        
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
