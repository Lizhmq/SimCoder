# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:47:00 2022

@author: DrLC
"""

import os
from tree_sitter import Language, Parser

def load_parser(lib_path="../dependency/tree-sitter-python",
                so_path="../dependency/tree-sitter-python-language.so",
                language="python"):
    
    # Need to clone the libarary first.
    # For instance, git clone https://github.com/tree-sitter/tree-sitter-python
    
    if not os.path.isfile(so_path):
        Language.build_library(so_path, [lib_path])
    parser = Parser()
    parser.set_language(Language(so_path, language))
    return parser

def obtain_token(node, _bytes):
    
    ret = []
    if len(node.children) <= 0 or node.type == 'string':
        tmp = str(_bytes[node.start_byte: node.end_byte], "latin1")
        tmp = tmp.strip()
        if len(tmp):
            return [tmp]
        else:
            return []
    for c in node.children:
        ret += obtain_token(c, _bytes)
    return ret

def traverse_tree(node, _bytes, layer=0):
    
    if len(node.children) <= 0:
        print ("\t" * layer +  node.type + "\t" + str(_bytes[node.start_byte: node.end_byte], "latin1"))
    else:
        print ("\t" * layer + node.type)
    for c in node.children:
        traverse_tree(c, _bytes, layer + 1)

if __name__ == "__main__":

    parser = load_parser()
    _bytes = bytes('''
    class ManifestedStaticURLGenerator(object):
        """ Adapter to generate static URLs using an `Assetgen`_ manifest file.
          
          _`Assetgen`: http://pypi.python.org/pypi/assetgen
          
        """
        
        adapts(IRequest, ISettings)
        implements(IStaticURLGenerator)
        
        def __init__(self, request, settings):
            self._dev = settings.get('dev', False)
            self._host = settings.get('static_host', request.host)
            self._static_url_prefix = settings['static_url_prefix']
            self._manifest = settings['assetgen_manifest']
            self._subdomains = cycle(
                settings.get('static_subdomains', '12345')
            self._val = 12345
            self._str = "12345"
            )
    ''', "latin1") 
    tree = parser.parse(_bytes)
    print (obtain_token(tree.root_node, _bytes))
    # traverse_tree(tree.root_node, _bytes)