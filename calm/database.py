#coding:utf-8

from .config_loader import load_config
import os,sqlite3
from functools import reduce
from itertools import islice, count

####################################################
## sqlite helper functions
####################################################

"""
Wrapper for easily interfacing with sqlite databases.
Easily instantiate a db from a configuration directory, which has a single .yml or .json for each table
specifying the schema as follows:
    {"schema": [ [col_name1, type1], [col_name2, type2], ... ],
     "keys": [key_col1, key_col2, ... ],
     "index": [index_col1, index_col2, ... ]
     }
with the name for the table coming from the filename.
Legal types are from sqlite: {'TEXT','INTEGER','REAL','BLOB'}
You can just execute init_sqlite_db("path/to/database.db", read_schema("path/to/schema/folder"))
to create the db, and then use insert_rows 
"""

legalConfigExtensions = {'.json','.yml'}
sqlite_types = {'TEXT','INTEGER','REAL','BLOB'}




def init_sqlite_db(db_path,schema,init_index=False,index_suffix=''):
    con = sqlite3.connect(db_path)
    
    for table, table_info in schema.items():
        create_table(con,init_index=init_index,index_suffix=index_suffix,**table_info)
        
    con.close()
    

def read_schema(schema_dir,extensions=['.yml','.json'],legal_types=sqlite_types):
    tables = [name for name in os.listdir(schema_dir) if os.path.splitext(name)[1] in extensions]
    db_schema = {}
    for filename in tables:
        table_info = read_table_config(os.path.join(schema_dir,filename),legal_types=legal_types)
        db_schema[table_info['name']] = table_info
        
    return db_schema


def read_table_config(table_path,legal_types=sqlite_types):
    table_info = load_config(table_path)
    table = table_info.get('name',os.path.splitext(os.path.basename(table_path))[0])
    keys = table_info.get('keys',None)
    index = table_info.get('index',None)
    schema = reduce(lambda l1,l2: l1+l2, table_info['schema'],[])
    datatypes = schema[1::2]
    fields = schema[0::2]
    for key in keys:
        if key not in fields:
            raise ValueError("Error in table config {}: key {} not in schema".format(filename,key))
    if index:
        for field in index:
            if field not in fields:
                raise ValueError("Error in table config {}: index field {} not in schema".format(filename,field))
    for datatype in datatypes:
        if datatype not in legal_types:
            raise ValueError("Error in table config {}: datatype {} not in legal_types: {}".format(filename,datatype,legal_types))
    
    table_info.update({'name':table,'fields':fields,'types':datatypes})
    
    return table_info

        
def create_table(con,init_index=False,index_suffix='',**table_info):
    """
    con is a sqlite3 database connection object
    init_index: boolean.  Create an index now if one is specified in table_info?
        it may be better to wait until the table is built, and then call init_index(con,**table_info)
    table_info can be passed as a dict of keyword args.
        Most robust method is to read this from a table config with read_table_config()
        Implicit keyword args are:
        - fields: list of field names in the table
        - keys: list of field names to serve as the primary key
        - types: list of datatypes as strings
        - index: list of field names to be used optionally for creating an index for fast lookup
    Create the table if it doesn't exist, with the specified schema.
    If the table exists, update with new columns implied by the schema.
    """
    keys = table_info.get('keys',None)
    schema = table_info['schema']
    cols = dict(schema)
    schemalist = reduce(lambda l1,l2: l1 + l2, schema, [])
    name = table_info['name']
    
    # create it
    if keys is None:
        command = "create table if not exists {} ({})".format(name, ", ".join(("{} {}",)*(len(schema)//2))).format(*schemalist)
    else:
        command = ("create table if not exists {} (" + "{} {}, "*(len(schema)//2) + "PRIMARY KEY (" + ','.join(keys) + ") )").format(name,*schemalist)
    cur = con.cursor()
    cur.execute(command)
    con.commit()
    cur.close()
    
    # update it
    cur = con.execute('select * from {} limit 0'.format(name))
    current_cols = set(desc[0] for desc in cur.description)
    new_cols = set(cols).difference(current_cols)
    # add new cols
    if len(new_cols) > 0:
        command = "alter table " + name + " add column {} {}"
        cur = con.cursor()
        for tup in schema:
            if tup[0] in new_cols:
                cur.execute(command.format(*tup))
        con.commit()
        cur.close()
    
    if init_index:
        if 'indices' not in table_info:
            print("Warning: init_index is True but no index is specified for table {}".format(name))
        else:
            create_index(con,index_suffix,**table_info)


def create_index(con,index_suffix='',**table_info):
    indices = table_info['indices']
    name = table_info['name']
    cur = con.cursor()
    for index_name, index in indices.items():
        command = "create index if not exists {}_{}{} on {}(".format(name,index_name,index_suffix,name) + ','.join(index) + ")"
        cur.execute(command)
    con.commit()
    cur.close()
 
    
def insert_rows(cur,table,data,fields,how='replace'):
    """
    cur: a sqlite3 cursor object.
    fields: the columns you're inserting data into
    data: an iterable of tuples of length the number of columns inserting into, of types compatible with
      the columns referred to by fields.
    how: a sqlite keyword specifying what to do on insert failure - one of ('replace','rollback','abort','fail','ignore')
      default is 'replace'.
    """
    command = ("insert or {} into {} ({}) VALUES ({})").format(how,table,','.join(fields),','.join(["?"]*len(fields)))
    cur.executemany(command,data)


def update_rows(cur,table,data,fields,key,add_new=True):
    """
    cur: a sqlite3 cursor object.
    fields: the columns you're inserting data into
    data: an iterable of tuples of length the number of columns inserting into, of types compatible with
      the columns referred to by fields.
    key: a column or list of columns (contained in fields) serving as a key for where to insert values.
    how: a sqlite keyword specifying what to do on insert failure - one of ('replace','rollback','abort','fail','ignore')
      default is 'replace'.
    """
    if type(key) is str:
        key = (key,)
    field_set = set(fields)
    if any(k not in field_set for k in key):
        raise KeyError("keys must be in the table fields")
    
    key_indices = dict((k,i) for i,k in enumerate(fields) if k in key)
    key_indices = tuple(key_indices[k] for k in key)
    def get_keys(t):
        return tuple(t[i] for i in key_indices)
    
    field_str = ', '.join("{}=?".format(f) for f in fields)
    key_str = ', '.join("{}=?".format(k) for k in key)
    
    command = ("update {} SET {} where {}").format(table,field_str,key_str)
    
    if add_new:
        insert_rows(cur,table,data,fields,how='ignore')
    cur.executemany(command,(d+get_keys(d) for d in data))


class Grouped:
    """
    iterate over an iterable in small chunks
    """
    def __init__(self,iterable,size):
        self.iterator = iter(iterable)
        self.size = size
    
    def __next__(self):
        slice = tuple(islice(self.iterator, self.size))
        if len(slice) > 0:
            return slice
        else:
            raise StopIteration
    
    def __iter__(self):
        return self


def update_rows_from_dicts_batched(con, table, key, data, batch_size = 100, max_batches = None, verbose = False):
    """
    take an iterator of dicts (data),
    group into batches of specified size,
    insert or update rows in the specified table using the specified key.
    commit on every batch_size inserts.
    """
    if max_batches is None:
        counter = count()
    else:
        counter = range(num_test_batches)
        
    for i,batch in zip(counter, Grouped(data, batch_size)):
        cur = con.cursor()
        for row in batch:
            # this is a dict of column name: value
            fields,values = zip(*row.items())
            try:
                update_rows(cur=cur, table=table, data=(values,), fields=fields, key=key, add_new=True)
            except Exception as e:
                if verbose:
                    print("error {} while inserting values {} for columns {}\n".format(e, values, fields))
        con.commit()
        cur.close()

