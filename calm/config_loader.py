import os
import re
import logging
import argparse
import time
from importlib import import_module


legalConfigExtensions = ('.json','.yml','.py')

# filter for comparing keywords
spaces = re.compile('[-_\s]')

# A handy arg parser allowing basic standard arguments: log directory,
# config file or directory, and verbose flag

def BasicParser():
    basicParser = argparse.ArgumentParser(add_help=True)

    basicParser.add_argument('--config','--config-path',type=str,
            metavar='CONFIG_PATH',help='path to configuration file or directory')
    basicParser.add_argument('--log',type=str,metavar='LOG_DIRECTORY',default='/tmp',
            help='path to the directory where logs will be stored')
    basicParser.add_argument('-v','--verbose',action='store_true',
            help='print verbose status messages?')
    
    return basicParser


class BasicLogger:
    def __init__(self,logDir=None,name=__name__,date_format="%Y-%m-%d_%H:%M",level='INFO',print_to_screen=True):
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.__dict__[level])
        self.error = self.log.error
        self.exception = self.log.exception
        self.warn = self.log.warn
        self.info = self.log.info
        self.debug = self.log.debug
        
        if logDir is not None:
            self.log_path = os.path.join(logDir,name+"_"+time.strftime(date_format)+'.log')
            handler = logging.FileHandler(self.log_path)
            handler.setLevel(logging.__dict__[level])
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
        
        if print_to_screen:
            printer = logging.StreamHandler()
            printer.setLevel(logging.__dict__[level])
            printer.setFormatter(formatter)
            self.log.addHandler(printer)


def get_name(synonym,thesaurus,remove=spaces):
    """
    thesaurus is a dict of the form
        {keyword:{synonym1,synonym2,...}...}
    Finds the correct keyword given a synonym.  Synsets are assumed to be mutually disjoint,
    so that only one keyword can be returned.
    remove is a regex specifying a pattern to be removed from the synonym, such as whitespace.
    """
    
    if remove:
        synonym = re.sub(remove,'',synonym)

    synonym=synonym.lower().strip()

    names = [name for name in thesaurus if synonym in thesaurus[name] or synonym==name]

    if len(names)==0:
        raise ValueError("Unrecognized keyword: {}".format(synonym))
    elif len(names) > 1:
        raise ValueError("Invalid thesaurus- overlapping synonyms for keywords: {}".format(names))
    else:
        return names[0]


def clean_args(raw_args,thesaurus,remove=spaces):
    """
    Replaces keys in raw_args (a dict)  with their canonical synonyms from thesaurus.
    Useful for standardizing kwargs from e.g. a config file or user input before passing to a function.
    thesaurus is a dict of the form
        {keyword:{synonym1,synonym2,...}...}
    """
    args = {}
    for arg in raw_args:
        args[get_name(arg,thesaurus,remove)]=raw_args[arg]
    return args



def load_config(path, module_to_dict=True):
    """
    loads configuration from .json, .yml, and .py.
    always returns a dict, even in the .py case by default; it's up to you how you'd like to
    handle the result, e.g. as-is or using  globals.update(config) to get direct
    variable name access (only works if the result is a dict, which may not be the
    case for example with some .json files defining list structures at the top level.
    if module_to_dict is True (default), python modules are unpacked into an explicit dict.
    """

    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[-1]
    if ext not in legalConfigExtensions:
        raise ValueError("{} is not a currently supported extension."+
                        "Supported file types are: {}".format(ext,legalConfigExtensions))

    if ext=='.yml':
        from yaml import safe_load as load_yaml
        with open(path,'r') as infile:
            config = load_yaml(infile)
    elif ext=='.json':
        from json import load as load_json
        with open(path,'r') as infile:
            config = load_json(infile)
    elif ext=='.py':
        directory = os.path.split(path)[0]
        module = os.path.splitext(filename)[0]
        cwd = os.getcwd()
        if directory != '':
            os.chdir(directory)
        config_module = import_module(module)
        os.chdir(cwd)
        if module_to_dict:
            names = [name for name in dir(config_module) if not name.startswith('__')]
            config = dict()
            for name in names:
                config[name] = config_module.__dict__[name]
        else:
            config = config_module

    return config


def load_config_dir(path,extensions=['.yml','.py','.json'],module_to_dict=True,force_dict=True):
    """
    if more than one file or force_dict==True, returns a dict 
        {config_file_basename:config_object(dict or list),...}
    with a key for every legal file (by mime type) in directory.
    else returns a single config object (dict or list) coming from the single config file.
    """

    if os.path.isdir(path):
        config_files = os.listdir(path)
    else:
        config_files = [os.path.split(path)[-1]]
        path = os.path.split(path)[0]

    config_files = [name for name in config_files if os.path.splitext(name)[-1] in extensions]

    if len(config_files) > 1 or force_dict:
        config = dict()
        for config_file in config_files:
            name = os.path.splitext(config_file)[0]
            new_config = load_config(os.path.join(path,config_file))
            config[name] = new_config
    else:
        config = load_config(os.path.join(path,config_files[0]),module_to_dict)

    return config

